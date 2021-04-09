################################################
#
# Semi-Supervised Discrete VAE (SSDVAE) - Main Module
# Code for the main module of the Dag VAE
#
################################################
import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from EncDec import Encoder, Decoder, Attention, fix_enc_hidden, gather_last
from  torch import distributions
import DAG
import torch.nn.functional as F
from Beam import Beam
import data_utils
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, TUP_TOK

class SSDVAE(nn.Module):
    def __init__(self, emb_size, hsize, vocab, latents, cell_type="GRU", layers=2, bidir=True, pretrained=True, use_cuda=True, dropout=0.10,frame_max=None,template=None,latent_dim=None,latent_emb_dim=None,verb_max_idx=None):
        """
        Args:
            emb_size (int) : size of input word embeddings
            hsize (int or tuple) : size of the hidden state (for one direction of encoder). If this is an integer then it is assumed
            to be the size for the encoder, and decoder is set the same. If a Tuple, then it should contain (encoder size, dec size)
            latents (LatentNode) : The root of a latent node tree (Note: Size of latent embedding dims should be 2*hsize if bidir!)
            layers (int) : layers for encoder and decoder
            vocab (Vocab object)
            bidir (bool) : use bidirectional encoder?
            cell_type (str) : 'LSTM' or 'GRU'
            sos_idx (int) : id of the start of sentence token
            latent_vocab_embedding: [frame_max, vocab_size]  defined for p(w|h,f) (reconstruction)
                                    each clause has a frame (frame<= frame_max) we map each of them
                                    to a vocab token
        """
        super(SSDVAE, self).__init__()

        self.embd_size=emb_size
        self.vocab = vocab
        self.vocab_size=len(vocab.stoi.keys())
        self.cell_type = cell_type
        self.layers = layers
        self.bidir=bidir
        self.sos_idx = self.vocab.stoi[SOS_TOK]
        self.eos_idx = self.vocab.stoi[EOS_TOK]
        self.pad_idx = self.vocab.stoi[PAD_TOK]
        self.tup_idx = self.vocab.stoi[TUP_TOK]
        self.latent_root = latents
        self.latent_dim = latent_dim  #Num Frames
        print('SSDVAE latent_dim: ',self.latent_dim)
        self.latent_emb_dim = latent_emb_dim
        self.frame_max=frame_max
        self.template = template
        self.latents=latents
        self.use_cuda = use_cuda
        self.verb_max_idx = verb_max_idx

        if isinstance(hsize, tuple):
            self.enc_hsize, self.dec_hsize = hsize
        elif bidir:
            self.enc_hsize = hsize
            self.dec_hsize = 2*hsize
        else:
            self.enc_hsize = hsize
            self.dec_hsize = hsize
        in_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx)
        out_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx)
        self.template_to_frame = nn.Linear(self.template, self.frame_max,bias=False)
        self.template_to_vocab = nn.Linear(self.frame_max, self.vocab_size,bias=False)
        self.theta_layer=nn.Linear(self.layers*self.enc_hsize,self.template)

        if pretrained:
            print("Using Pretrained")
            in_embedding.weight.data = vocab.vectors
            out_embedding.weight.data = vocab.vectors


        self.encoder = Encoder(self.embd_size, self.enc_hsize, in_embedding, self.cell_type, self.layers, self.bidir, use_cuda=use_cuda)
        self.decoder = Decoder(self.embd_size, self.dec_hsize,self.vocab_size,out_embedding, self.cell_type, self.layers, attn_dim=(self.latent_dim, self.dec_hsize), use_cuda=use_cuda, dropout=dropout)



        self.logits_out= nn.Linear(self.dec_hsize, self.vocab_size) #Weights to calculate logits, out [batch, vocab_size]
        self.latent_in = nn.Linear(self.latent_dim, self.layers*self.dec_hsize) #Compute the query for the latents from the last encoder output vector
        if use_cuda:
            self.decoder = self.decoder.cuda()
            self.encoder = self.encoder.cuda()
            self.logits_out = self.logits_out.cuda()
            self.latent_in = self.latent_in.cuda()
            self.theta_layer = self.theta_layer.cuda()
            self.template_to_frame = self.template_to_frame.cuda()
            self.template_to_vocab = self.template_to_vocab.cuda()

        else:
            self.decoder = self.decoder
            self.encoder = self.encoder
            self.logits_out = self.logits_out
            self.latent_in = self.latent_in
            self.theta_layer = self.theta_layer

    def set_use_cuda(self, value):
        self.use_cuda = value
        self.encoder.use_cuda = value
        self.decoder.use_cuda = value
        self.decoder.attention.use_cuda = value
        self.latent_root.set_use_cuda(value)


    def forward(self, input, seq_lens,f_vals=None,beam_size=-1, str_out=False, max_len_decode=50, min_len_decode=0, n_best=1, encode_only=False):
        batch_size = input.size(0)
        if str_out: #use batch size 1 if trying to get actual output
            assert batch_size == 1

        # INIT THE ENCODER
        ehidden = self.encoder.initHidden(batch_size)
        enc_output, ehidden = self.encoder(input, ehidden, seq_lens)
        enc_theta = self.theta_layer(enc_output).mean(1) #[batch_size,template]
        p_theta_sampled = F.softmax(enc_theta,-1).cuda()
        template_input = F.tanh(self.template_to_frame(p_theta_sampled))
        self.template_decode_input = self.template_to_vocab(template_input)

        if self.use_cuda:
            enc_output_avg = torch.sum(enc_output, dim=1) / Variable(seq_lens.view(-1, 1).type(torch.FloatTensor).cuda())
        else:
            enc_output_avg = torch.sum(enc_output, dim=1) / Variable(seq_lens.view(-1, 1).type(torch.FloatTensor))
        initial_query = enc_output_avg
        latent_values, diffs,latent_embs,q_log_q ,frames_to_frames, frame_classifier, scores = self.latent_root.forward(enc_output, seq_lens, initial_query,f_vals,template_input=template_input) #[batch, num_clauses, num_frames]
        self.scores=scores
        self.latent_gumbels = latent_values
        self.frames_to_frames = frames_to_frames
        self.frame_classifier = frame_classifier


        self.q_log_q=q_log_q
        top_level = latent_embs[:, 0, :]
        dhidden = torch.nn.functional.tanh(self.latent_in(top_level).view(self.layers, batch_size, self.dec_hsize))

        if encode_only:
            if self.use_cuda:
                self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)).cuda())
            else:
                self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))
            return dhidden, latent_embs

        if str_out:
            if beam_size <=0:
                # GREEDY Decoding
                if self.use_cuda:
                    self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim).cuda())) #initialize the input feed 0
                else:
                    self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))

                return self.greedy_decode(input, dhidden, latent_embs, max_len_decode)
            else:
                # BEAM Decoding
                return self.beam_decode(input, dhidden, latent_embs, beam_size, max_len_decode, min_len_decode=min_len_decode)


        # This is for TRAINING, use teacher forcing
        if self.use_cuda:
            self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim).cuda())) #initialize the input feed 0
        else:
            self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))

        return self.train(input, batch_size, dhidden, latent_embs, diffs)


    def train(self, input, batch_size, dhidden, latent_embs, diffs, return_hid=False, use_eos=False):

        dec_outputs = []
        logits = []
        input_size = input.size(1) #Dont need to process last since no eos

        for i in range(input_size):
            #Choose input for this step
            if i == 0:
                tens = torch.LongTensor(input.shape[0]).zero_() + self.sos_idx
                if self.use_cuda:
                    dec_input = Variable(tens.cuda()) #Decoder input init with sos
                else:
                    dec_input = Variable(tens)
            else:
                dec_input = input[:, i-1]
            dec_output, dhidden ,logit , frame_to_vocab = self.decoder(dec_input, dhidden,latent_embs,self.template_decode_input)


            dec_outputs += [dec_output]
            logits += [logit]

        dec_outputs = torch.stack(dec_outputs, dim=0) 
        logits = torch.stack(logits, dim=0) 
        self.logits=logits
        self.frame_to_vocab=frame_to_vocab
        if return_hid:
            return latent_embs, self.latent_root, dhidden, dec_outputs 
        else:
            self.decoder.reset_feed_() 
            return latent_embs, self.latent_root, diffs, dec_outputs 


    def greedy_decode(self, input, dhidden, latent_values, max_len_decode):

        outputs = []
        dec_input = Variable(torch.LongTensor(input.shape[0]).zero_() + self.sos_idx)
        prev_output = Variable(torch.LongTensor(input.shape[0]).zero_() + self.sos_idx)

        if self.decoder.input_feed is None:
            if self.use_cuda:
                self.decoder.init_feed_(Variable(torch.zeros(1, self.decoder.attn_dim).cuda())) #initialize the input feed 0
            else:
                self.decoder.init_feed_(Variable(torch.zeros(1, self.decoder.attn_dim)))

        for i in range(max_len_decode):
            if i == 0: #first input is SOS (start of sentence)
                dec_input = Variable(torch.LongTensor(input.shape[0]).zero_() + self.sos_idx)
            else:
                dec_input = prev_output

            dec_output, dhidden = self.decoder(dec_input, dhidden,latent_values)

            logits = self.logits_out(dec_output)

            probs = F.log_softmax(logits, dim=1)
            top_vals, top_inds = probs.topk(1, dim=1)

            outputs.append(top_inds.squeeze().data[0])
            prev_output = top_inds

            if top_inds.squeeze().data[0] == self.eos_idx:
                break

        self.decoder.reset_feed_() #reset input feed so pytorch correctly cleans graph
        return outputs


    def beam_decode(self, input, dhidden, latent_values, beam_size, max_len_decode, n_best=1, use_constraints=True, min_len_decode=0, init_beam=False):
        batch_size = 1
        assert beam_size >= n_best, "n_best cannot be greater than beam_size."

        def var(a): return Variable(a, volatile=True)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        def beam_update(e, idx, positions, beam_size):
            sizes = e.size() # [1, beam_size, hidden_size]
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            indexed_before = sent_states.data.index_select(1, positions)
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))
            indexed_after = sent_states.data.index_select(1, positions)


        dhidden = dhidden.repeat(1, beam_size, 1)
        latent_values = latent_values.repeat(beam_size, 1, 1)

        if init_beam: #Init with the current input feed
            self.decoder.input_feed = self.decoder.input_feed.repeat(beam_size, 1)
        else:
            # init with beam_size zero
            if self.use_cuda:
                self.decoder.init_feed_(var(torch.zeros(beam_size, self.decoder.attn_dim).cuda())) #initialize the input feed 0
            else:
                self.decoder.init_feed_(var(torch.zeros(beam_size, self.decoder.attn_dim)))


        # 2. beam object as we have batch_size 1 during decoding
        beam = [Beam(beam_size, n_best=n_best,
                    cuda=self.use_cuda,
                    pad=1,
                    eos=self.eos_idx,
                    bos=self.sos_idx,
                    min_length=10)]

        if init_beam: #if init_beam is true, then input will be the initial input to init beam with
            for b in beam:
                b.next_ys[0][0] = np.asscalar(input.data.numpy()[0])

        verb_list = [[]]*beam_size #for constraints

        # run the decoder to generate the sequence
        for i in range(max_len_decode):

            # one all beams have EOS break
            if all((b.done() for b in beam)):
                break

            # No need to explicitly set the input to previous output - beam advance does it. Make sure.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                                  .t().contiguous().view(-1)) #[beam_size]
            # Tested that the last output is the input in the next time step.
            #print("STEP {}".format(i))
            #print("INPUT: {}".format(inp.data))

            # Run one step of the decoder
            # dec_out: beam x rnn_size

            dec_output, dhidden = self.decoder(inp, dhidden, latent_values)
            # [1, beam_size, hidden_size]
            dec_output = torch.unsqueeze(dec_output, 0)
            logits = self.logits_out(dec_output)
            probs = F.log_softmax(logits, dim=2).data
            out = unbottle(probs) # [beam_size, 1, vocab_size]
            out.log()

            # Advance each beam.

            for j, b in enumerate(beam):
                if use_constraints:
                    b.advance(ge.schema_constraint(out[:,j], b.next_ys[-1], verb_list, min_len_decode=min_len_decode, step=i, eos_idx=self.eos_idx))
                else:
                    b.advance(out[:, j])

                # advance hidden state and input feed  accordingly
                beam_update(dhidden, j, b.get_current_origin(), beam_size)
                beam_update(dec_output, j, b.get_current_origin(), beam_size)

                if use_constraints:
                    verb_list = ge.update_verb_list(verb_list, b, self.tup_idx) #update list of currently used verbs

            self.decoder.input_feed = dec_output.squeeze(dim=0) # update input feed for the next step.

        # extract sentences (token ids) from beam and return
        ret = self._from_beam(beam, n_best=n_best)[0][0] # best hyp

        self.decoder.reset_feed_() #reset input feed so pytorch correctly cleans graph
        return ret

    def _from_beam(self, beam, n_best=1):
        ret = {"predictions": [],
               "scores": []}
        for b in beam: # Only 1 beam object.
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times, k)
                hyps.append(hyp)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)

        return ret['predictions']


    def decode(self, input, impute=None):
        outputs = []
        my_lats = [self.latent_root]
        child1 = self.latent_root.children[0]
        child2 = child1.children[0]
        child3 = child2.children[0]
        child4 = child3.children[0]
        my_lats.append(child1)
        my_lats.append(child2)
        my_lats.append(child3)
        my_lats.append(child4)

        latent_list = []
        for i, ind in enumerate(input):
            print("decode i: {}.".format(i))
            if impute is not None and i == 0:
                latent_list.append(impute[1]*my_lats[i].embeddings(Variable(torch.LongTensor([ind])))
                        + (1-impute[1])*my_lats[i].embeddings(Variable(torch.LongTensor([impute[0]]))))
            else:
                latent_list.append(my_lats[i].embeddings(Variable(torch.LongTensor([ind]))))
        latent_list = torch.stack(latent_list, 0)
        latent_values = latent_list.transpose(0,1)
        top_level = latent_values[:, 0, :]
        dhidden = self.latent_in(top_level).view(self.layers,1, self.dec_hsize)
        #DECODE
        outputs = self.beam_decode(torch.Tensor(1,1), dhidden, latent_values, 10, 50)


        self.decoder.reset_feed_() #reset input feed so pytorch correctly cleans graph
        return outputs
