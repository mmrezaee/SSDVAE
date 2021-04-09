##################################
#   The DAG structures for the latent space
#   Currently this is just a tree, sorry for the false advertising (can be extended to a dag latter on)
##################################
import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from EncDec import Encoder, Decoder, Attention, gather_last
from  torch import distributions
import torch.nn.functional as Func
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical as RelaxCat 


class LatentNode(nn.Module):
    'a node in the latent dag graph, represents a latent variable'
    def __init__(self, num_Frames, dim, nodeid="0", embeddings=None, use_attn=True, use_cuda=True, nohier_mode=False,train_mode=True):
        """
        Args
            num_Frames (int) : number of (frames) latent categorical values this can take on (and thus, the # of embeddings)
            dim (int tuple) :  (query dimension, encoder input (memory) dimension, latent embedding dimension (output))
            nodeid (str) : an optional id for naming the node
            embeddings (nn.Embeddings) : Pass these if you want to create the embeddings, else just go with default
            nohier_mode (bool) : Run the NOHIER model instead
        """
        super(LatentNode, self).__init__()
        self.children = []  #list of LatentNodes
        self.parents = []
        self.use_cuda = use_cuda
        self.value = None
        self.diffs = None
        self.index_value = None #Index is the indices into the embedding of the above
        self.nohier = nohier_mode
        self.LogSoftmax=nn.LogSoftmax()


        self.tau= 0.5
        if use_attn:
            if use_cuda:
                self.attn = Attention(dim,is_latent=True).cuda()                
            else:
                self.attn = Attention(dim, use_cuda=False,is_latent=True)
        else:
            self.attn=None

        self.nodeid=nodeid
        self.num_Frames = num_Frames
        self.dim = dim[2] #dimension of the embeddings for the latent nodes
        print('dim: ',dim)
        print('self.dim: ',dim[2])
        self.all_dims = dim

        if embeddings is not None:
            if use_cuda:
                self.embeddings = embeddings.cuda()
            else:
                self.embeddings = embeddings
        else:
            if use_cuda:
                self.embeddings = nn.Embedding(F, self.dim).cuda()
            else:
                self.embeddings = nn.Embedding(F, self.dim)

            self.embeddings.weight.data = torch.nn.init.xavier_uniform(self.embeddings.weight.data)

        #Don't forget to initialize weights

    def isroot(self):
        return self.parents == []
    def isleaf(self):
        return self.children == []

    def add_child_(self, child):
        """
        Args
            child (LatenNode) : Latent node to add as a child to this node
        """
        child.parents.append(self)
        self.children.append(child)
        self.add_module(child.nodeid, child) #This is important so the children acutally get updated

    def prune_(self):
        """
        Prune embeddings for self and children
        """
        self.embeddings.weight = nn.Parameter(prune_latents(self.embeddings.weight.data, 1))
        self.num_Frames = self.embeddings.weight.data.shape[0]
        for child in self.children:
            child.prune_()

    def zero_attn_grads(self):
        """
        zero out the attn grads so they are not updated twice
        """
        self.attn.zero_grad()
        for child in self.children:
            child.zero_attn_grads()


    def set_use_cuda(self, value):
        self.use_cuda = value
        self.attn.use_cuda = value
        for child in self.children:
            child.set_use_cuda(value)

    def frames_onehot(self,f_vals_clause):
        """
        f_vals_clause: [batch]
        (Each time one of the frames is called)
        """

        frames_clause_=f_vals_clause.unsqueeze(0).unsqueeze(-1)
        # print("frames_clause_: ",frames_clause_)
        # print("frames_clause_size: ",frames_clause_.size())
        if self.use_cuda:
            one_hot = torch.zeros(frames_clause_.size()[0],frames_clause_.size()[1],self.num_Frames).cuda()
        else:
            one_hot = torch.zeros(frames_clause_.size()[0],frames_clause_.size()[1],self.num_Frames)
        one_hot.scatter_(2, frames_clause_, 1)
        # print("one_hot_after: ",one_hot)
        # print("one_hot_after: ",one_hot.size())
        return one_hot  

    def infer_(self, input_memory, input_lens, init_query=None,f_vals=None,template_input=None):
        """
        Calculate the current value of Z for this variable (deterministic sample),
        given the values of parents and the input memory, then update this variables value
        Args:
            memory (FloatTensor, [batch, seq_len, dim]) : The input encoder states (what is attended to)
            Input lens is a [batch] size Tensor with the lengths of inputs for each batch
            init_query (an initial query for the root)

            input_memory: [batch_size,seq_len,num_hidden]
            input_lens: [batch_size] it contains the length (just a number) 
                        of each input and will be used for masking            

            init_query: [batch_size,num_hidden] the average of hiddens from observed tokens
            prev_latent: [batch_size,num_Frames] these are the parameters of gumbel_softmax to sample the next frame
        """

        if not self.isroot() and not self.nohier:  #if we are a child node AND we are not running in nohier mode
            which_row=int(self.nodeid.split('_')[-1])
            f_val_child=f_vals[:,which_row]
            l_o = Variable((f_val_child>0).view((-1,1))).float()
            l_c = Variable((f_val_child>2).view((-1,1))).float()
            fval_one_hot=self.frames_onehot(f_val_child)        
            prev_latent = self.parents[0].value #propogate decoder loss back through attn and any previous attns
            V, scores, frame_to_frame, vocab_to_frame = self.attn(prev_latent, input_memory, input_lens) #unnormalized gumbel parameter
            V += template_input

        else:
            root_row = 0
            f_val_root = f_vals[:,root_row]
            # l_o = Variable(f_val_root>0).float()
            l_o = Variable((f_val_root>0).view((-1,1))).float()
            l_c = Variable((f_val_root>2).view((-1,1))).float()            
            fval_one_hot = self.frames_onehot(f_val_root)
            V, scores , frame_to_frame, vocab_to_frame= self.attn(init_query, input_memory, input_lens) #unnormalized gumbel parameter
            V += template_input

        V_raw= V
        V=(1-l_o)*V+(l_o*torch.norm(V)* fval_one_hot.squeeze(0))

        gumbel_samples = torch.nn.functional.gumbel_softmax(logits=V,tau=torch.tensor([self.tau]).cuda())
        self.frame_classifier = 0.1*l_c*self.LogSoftmax(V_raw)*fval_one_hot.squeeze(0)
        frame_to_frame_probs = frame_to_frame.squeeze()
        self.frame_to_frame_probs = frame_to_frame_probs
        probs=0.1*(1-l_o)*Func.softmax(V,-1)

        batch_size = V.shape[0]
        self.scores=vocab_to_frame

        if self.isroot() or self.nohier:

            self.value=torch.mm(gumbel_samples,self.embeddings.weight)
            self.gumbel_samples=gumbel_samples #[batch,num_frames]
            self.q_log_q=(probs*torch.log(probs+1e-10)).sum(-1)#[batch]
            self.diffs =  (torch.zeros((1,1)).cuda(),torch.zeros((1,1)).cuda())

        else:
            self.value=torch.mm(gumbel_samples,self.embeddings.weight)
            self.diffs =  (torch.zeros((1,1)).cuda(),torch.zeros((1,1)).cuda(), torch.zeros((1,1)).cuda())
            self.gumbel_samples=gumbel_samples
            self.q_log_q=(probs*torch.log(probs+1e-10)).sum(-1)#[batch]






    def infer_all_(self, input_memory, input_lens, init_query=None,f_vals=None,template_input=None):
        """
        Call infer recusivly down the tree, starting from this node
        Args:
            memory (FloatTensor, [batch, seq_len, dim]) : The input encoder states (what is attended to)
            f_vals [batch,num_clauses]
        """
        self.infer_(input_memory, input_lens, init_query,f_vals,template_input=template_input)
        for idx_child,child in enumerate(self.children):
            child.infer_all_(input_memory, input_lens, init_query,f_vals,template_input=template_input)


    def forward(self, input_memory, input_lens, init_query,f_vals=None,template_input=None):
        """
        Input lens is a [batch] size Tensor with the lengths of inputs for each batch
        init_query: [batch_size,num_hidden] the average of hiddens from observed tokens
        input_memory: [batch_size,seq_len,num_hidden]
        input_lens: [batch_size] it contains the length (just a number) 
                    of each input and will be used for masking
        f_vals: observed (and not) frames [batch, num_clauses]
        """
        self.infer_all_(input_memory, input_lens, init_query,f_vals=f_vals,template_input=template_input)
        collected = self.collect()
        diffs = self.collect_diffs()
        embs = self.collect_embs()
        q_log_q=self.collect_q_log_q()
        frames_to_frames=self.collect_frame_to_frame()
        frame_classifier= self.collect_classifier()
        scores= self.collect_scores()
        self.reset_values()
        return collected, diffs,embs,q_log_q,frames_to_frames, frame_classifier,scores

    def collect_diffs(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, dim])
        """
        diff_list = [self.diffs]
        for child in self.children:
            diff_list += child.collect_diffs()

        return diff_list
    def collect_scores(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, dim])
        """
        scores = [self.scores]
        for child in self.children:
            scores += child.collect_scores()

        if self.isroot():
            return torch.stack(scores, dim=1)            
        return scores

    def collect_classifier(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, dim])
        """
        frame_classifier = [self.frame_classifier]
        for child in self.children:
            frame_classifier += child.collect_classifier()

        if self.isroot():
            return torch.stack(frame_classifier, dim=1)            
        return frame_classifier


    def collect_frame_to_frame(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, dim])
        """
        frame_to_frame_list = [self.frame_to_frame_probs]
        for child in self.children:
            frame_to_frame_list += child.collect_frame_to_frame()

        if self.isroot():
            return torch.stack(frame_to_frame_list, dim=1)            
        return frame_to_frame_list

    def collect_q_log_q(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, dim])
        """
        q_log_list = [self.q_log_q]
        for child in self.children:
            q_log_list += child.collect_q_log_q()

        if self.isroot():
            return torch.stack(q_log_list, dim=-1)
        else:
            return q_log_list

    def collect(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, latent_dim])
        """
        latent_list = [self.gumbel_samples]
        for child in self.children:
            latent_list += child.collect()

        if self.isroot():
            return torch.stack(latent_list, dim=1)
        else:
            return latent_list

    def collect_embs(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, dim])
        """
        emb_list = [self.value]
        for child in self.children:
            emb_list += child.collect_embs()
        if self.isroot():
            return torch.stack(emb_list, dim=1)
        else:
            return emb_list

    def reset_values(self):
        """
        Reset all of the values for each node (so
        that pytorch cleans up the Variables for the next round)
        """

        self.diffs = None
        self.value = None
        for child in self.children:
            child.reset_values()

    def set_nohier(self, value=False):
        """
        Set nohier attribute to false for this node and all children.
        This is for backwards compatibility with older versions
        """
        self.nohier = value
        for child in self.children:
            child.set_nohier(value)


def example_tree(num_Frames, all_dim,frame_max,padding_idx=None,use_cuda=True, nohier_mode=False):
    """
    An example function of building trees/dags to use in DAVAE
    num_Frames: num_latent (how many discrete values for each node? number of frames for our case)
    latent_dim (num_Frames): (the dimension of embedding vector representation for each frame, we use frame as well)
    all_dim : tuple (encoder dim, latent_dim)
    root_dim: (hidden,hidden,latent_dim)
    child: (latent_dim,hidden,latent_dim)
    all_dim : (hidden,latent_dim)
    frame_embedding: (frame_max,latent_dim) maps a frame to another frame
    """

                #Query dim  #Mem Dim   #Latent Dim
    # root_dim = (all_dim[0], all_dim[0], all_dim[1])
    root_dim = (all_dim[0], all_dim[0], frame_max)

    if nohier_mode:
        dim = root_dim
    else:
        dim = (all_dim[1], all_dim[0], frame_max)
        # dim = (all_dim[1], all_dim[0], all_dim[1])

    frame_embedding = nn.Embedding(frame_max, all_dim[1],padding_idx=padding_idx)
    print('frame_embedding: ',frame_embedding.weight.size())
    # q_theta_dist = distributions.Dirichlet(alpha)
    # def __init__(self, K, dim, nodeid="0", embeddings=None, use_attn=True, use_cuda=True, nohier_mode=False):
    
    root = LatentNode(num_Frames, root_dim, nodeid="ROOT",embeddings=frame_embedding,
                      use_cuda=use_cuda, nohier_mode=nohier_mode)

    child_F=num_Frames

    if nohier_mode:
        print("Using NOHIER")

    #THIS WORKS FINE (Use Xavier_normal)
    print("Using Linear Chain")
    i=1
    id_str = "Level_{}".format(i)
    child1= LatentNode(child_F, dim, nodeid=id_str, embeddings=frame_embedding,
                       use_cuda=use_cuda, nohier_mode=nohier_mode)

    i+=1
    id_str = "Level_{}".format(i)
    child2= LatentNode(child_F, dim, nodeid=id_str, embeddings=frame_embedding,
                       use_cuda=use_cuda, nohier_mode=nohier_mode)

    i+=1
    id_str = "Level_{}".format(i)
    child3= LatentNode(child_F, dim, nodeid=id_str, embeddings=frame_embedding,
                       use_cuda=use_cuda, nohier_mode=nohier_mode)

    i+=1
    id_str = "Level_{}".format(i)
    child4= LatentNode(child_F, dim, nodeid=id_str, embeddings=frame_embedding,
                       use_cuda=use_cuda, nohier_mode=nohier_mode)

    child3.add_child_(child4)
    child2.add_child_(child3)
    child1.add_child_(child2)
    root.add_child_(child1)

    return root

def prune_latents(embeddings, threshold_norm):
    """
    Remove any embeddings with norm less than threshold_norm (these were probably not updated in training)
    args
        embeddings (Tensor [num_Frames x dim])
        threshold_norm (int)
    """
    num_Frames = embeddings.shape[0]
    indices = torch.masked_select(torch.arange(num_Frames), torch.ge(torch.norm(embeddings, 2, dim=1), threshold_norm))
    return embeddings[indices.type(torch.LongTensor)]
