###########################################
# Model for generating samples from model
#
###########################################
import torch
import torch.nn as nn
from torchtext.data import Iterator as BatchIter
import argparse
import pickle
import numpy as np
import math
import time
from torch.autograd import Variable
import torch.nn.functional as F
import data_utils as du
import os
import random

import DAVAE
from DAG import example_tree
from EncDec import Encoder, Decoder, Attention, fix_enc_hidden
from masked_cross_entropy import masked_cross_entropy
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, TUP_TOK, MAX_EVAL_SEQ_LEN, MIN_EVAL_SEQ_LEN
from decode_utils import transform, get_tups, get_pred_events

def generate(args):
    """
    Use the trained model for decoding
    Args
        args (argparse.ArgumentParser)
    """
    if args.cuda and torch.cuda.is_available():
        device = 0
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        device = -1
        use_cuda = False
    else:
        device = -1
        use_cuda=False

    #Load the vocab
    # vocab = du.load_vocab(args.vocab)
    vocab , _ = du.load_vocab(args.vocab)
    vocab2 = du.load_vocab(args.frame_vocab_address,is_Frame=True)

    eos_id = vocab.stoi[EOS_TOK]
    pad_id = vocab.stoi[PAD_TOK]

    if args.ranking: # default is HARD one, the 'Inverse Narrative Cloze' in the paper
        dataset = du.NarrativeClozeDataset(args.valid_narr, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN, LM=False)
        # Batch size during decoding is set to 1
        batches = BatchIter(dataset, 1, sort_key=lambda x:len(x.actual), train=False, device=-1)
    else:
        # dataset = du.SentenceDataset(args.valid_data, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN, add_eos=False) #put in filter pred later
        dataset = du.SentenceDataset(path=args.valid_data,path2=args.valid_frames,vocab=vocab,vocab2=vocab2,num_clauses=args.num_clauses, add_eos=False,is_ref=True,obsv_prob=0.0,print_valid=True)
        # Batch size during decoding is set to 1
        batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, device=-1)

    data_len = len(dataset)

    #Create the model
    with open(args.load, 'rb') as fi:
        if not use_cuda:
            model = torch.load(fi, map_location=lambda storage, loc : storage)
        else:
            model = torch.load(fi, map_location=torch.device('cuda'))

    if not hasattr(model.latent_root, 'nohier'):
        model.latent_root.set_nohier(args.nohier) #for backwards compatibility

    model.decoder.eval()
    model.set_use_cuda(use_cuda)

    #For reconstruction
    if args.perplexity:
        print('calculating perplexity')
        loss = calc_perplexity(args, model, batches, vocab, data_len)
        NLL=loss
        PPL=np.exp(loss)
        print("Chain-NLL = {}".format(NLL))
        print("Chain-PPL = {}".format(PPL))
        return PPL
    elif args.schema:
        generate_from_seed(args, model, batches, vocab, data_len)
    elif args.ranking:
        ranked_acc=do_ranking(args, model, batches, vocab, vocab2, data_len, use_cuda)
        return ranked_acc
    else:
#        sample_outputs(model, vocab)
        reconstruct(args, model, batches, vocab)

#Inverse Narrative Cloze
def do_ranking(args, model, batches, vocab, vocab2, data_len, use_cuda):
    print("RANKING")
    ranked_acc = 0.0

    tup_idx = vocab.stoi[TUP_TOK]


    for iteration, bl in enumerate(batches):

        if (iteration+1)%25 == 0:
            print("iteration {}".format(iteration+1))

        all_texts = [bl.actual, bl.actual_tgt, bl.dist1, bl.dist1_tgt, bl.dist2, bl.dist2_tgt, bl.dist3, bl.dist3_tgt, bl.dist4, bl.dist4_tgt, bl.dist5, bl.dist5_tgt] # each is a tup

        assert len(all_texts) == 12, "12 = 6 * 2."

        all_texts_vars = []

        for tup in all_texts:
            all_texts_vars.append((Variable(tup[0], volatile=True), tup[1]))

        # will itetrate 2 at a time using iterator and next
        vars_iter = iter(all_texts_vars)
        # run the model for all 6 sentences
        pps = []

        first_tup = -1
        for i in range(bl.actual[0].shape[1]):
            if bl.actual[0][0, i] == tup_idx:
                first_tup = i
                break
        if first_tup == -1:
            print("WARNING: First TUP is -1")
        # print("bl.actual: ",bl.actual[0].size())
        src_tup = Variable(bl.actual[0][:, :first_tup+1].view(1, -1), volatile=True)
        src_lens = torch.LongTensor([src_tup.shape[1]])
        f_vals = torch.LongTensor([[0,2,2,2,2]])

        # print("f_vals: ",f_vals.size())

        if use_cuda:
            src_tup = src_tup.cuda()
            src_lens = src_lens.cuda()
            f_vals = f_vals.cuda()
        # print('src_tup: ',src_tup.size())
        dhidden, latent_values = model(src_tup, src_lens,f_vals=f_vals,encode_only=True)        
        latent_gumbels = model.latent_gumbels
        frames_infer = latent_gumbels[0,:,:].data.cpu()    
        # total_frames_infer=latent_gumbels.data.cpu()
        seed_infer_idx = torch.argmax(frames_infer, -1)[0]        
        # seed_meaning = [vocab.itos[int(item.numpy())] for item in src_tup.cpu().data[0]]
        # seed_frame_infer = vocab2.itos[int(seed_infer_idx.numpy())]
        # f_vals = torch.LongTensor([[seed_infer_idx,0,0,0,0]]).cuda()
        f_vals = torch.LongTensor([[seed_infer_idx,0,0,0,0]]).cuda()
        # print("seed_meaning: ",seed_meaning)
        # print('seed_frame_infer: ',seed_frame_infer)
        # Latent and hidden have been initialized with the first tuple
        for tup in vars_iter:
            ## INIT FEED AND DECODE before every sentence.
            if use_cuda:
                model.decoder.init_feed_(Variable(torch.zeros(1, model.decoder.attn_dim).cuda()))
            else:
                model.decoder.init_feed_(Variable(torch.zeros(1, model.decoder.attn_dim)))
            next_tup = next(vars_iter)
            # print("next_tup[0]: ",next_tup[0].size())
            # print("next_tup[1]: ",next_tup[1])
            cropped_input = next_tup[0][0,:24].view(1,-1).cuda()
            cropped_input_len = next_tup[1]-5
            # print("cropped_input: ",cropped_input.size())
            # print("cropped_input_len: ",cropped_input_len)
            # _, _, _, dec_outputs  = model(next_tup[0].cuda(), Variable(next_tup[1]).cuda(),f_vals=f_vals)
            _, _, _, dec_outputs  = model(cropped_input.cuda(), Variable(cropped_input_len).cuda(),f_vals=f_vals)
            logits = model.logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]                        
            # print("logits: ",logits.size())
            tup_latent_gumbels = model.latent_gumbels
            tup_frames_infer = tup_latent_gumbels[0,:,:].data.cpu()    
            tup_total_frames_infer=tup_latent_gumbels.data.cpu()
            tup_seed_infer_idx = torch.argmax(tup_frames_infer, -1)
            # print("next_tup_meaning: ",[vocab.itos[int(item.numpy())] for item in next_tup[0].cpu().data[0]])
            tup_seed_infer_idx_meaning = [vocab2.itos[int(item.numpy())] for item in tup_seed_infer_idx.cpu().data]
            # print("tup_seed_infer_idx_meaning: ",tup_seed_infer_idx_meaning)
            # logits = logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]
            # print("logits: ",logits.size())
            # print('next_tup[0]',next_tup[0].size())
            # print('next_tup[1]',next_tup[0].size())
            # nll = masked_cross_entropy(logits, next_tup[0].cuda(), Variable(next_tup[1]).cuda())
            nll = masked_cross_entropy(logits, cropped_input, Variable(cropped_input_len).cuda())
            #nll = calc_perplexity(args, model, tup[0], vocab, next_tup[0], next_tup[1], hidden)
            pp = torch.exp(nll)
            #print("NEG-LOSS {} PPL {}".format(nll.data[0], pp.data[0]))
            pps.append(pp.data.cpu().numpy())

        # low perplexity == top ranked sentence- correct answer is the first one of course
        assert len(pps) == 6, "6 targets."
        #print("\n")
        all_texts_str = [transform(text[0].data.numpy()[0], vocab.itos) for text in all_texts_vars]
        #print("ALL: {}".format(all_texts_str))
        min_index = np.argmin(pps)
        if min_index == 0:
            ranked_acc += 1

            #print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
            #print("CORRECT: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
        #else:
            # print the ones that are wrong
            #print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
            #print("WRONG: {}".format(transform(all_texts_vars[min_index+2][0].data.numpy()[0], vocab.itos)))

        if (iteration+1) == args.max_decode:
            print("Max decode reached. Exiting.")
            break

    ranked_acc /= (iteration+1) * 1/100 # multiplying to get percent
    print("Average acc(%): {}".format(ranked_acc))
    return ranked_acc


def calc_perplexity_avg_line(args, model, batches, vocab, data_len):
    total_loss = 0.0
    iters = 0
    for iteration, bl in enumerate(batches):
        print(iteration)
        batch, batch_lens = bl.text
        target, target_lens = bl.target
        if args.cuda:
            batch = Variable(batch.cuda(), volatile=True)
        else:
            batch = Variable(batch, volatile=True)

        _, _, _, dec_outputs  = model(batch, batch_lens)

        logits = model.logits_out(dec_outputs).cpu()

        logits = logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]

        ce_loss = masked_cross_entropy(logits, Variable(target), Variable(target_lens))
        total_loss = total_loss + ce_loss.data[0]

        iters += 1

    print(iters)
    print(data_len)

    return total_loss / data_len

def calc_perplexity(args, model, batches, vocab, data_len):
    total_loss = 0.0
    iters = 0
    total_words = 0
    for iteration, bl in enumerate(batches):
        print(iteration)
        batch, batch_lens = bl.text
        target, target_lens = bl.target
        f_vals,f_vals_lens = bl.frame

        if args.cuda:
            batch = Variable(batch.cuda(), volatile=True)
            f_vals = Variable(f_vals.cuda(), volatile=True)
        else:
            batch = Variable(batch, volatile=True)
            f_vals = Variable(f_vals.cuda())

        _, _, _, dec_outputs  = model(batch, batch_lens,f_vals=f_vals)
        logits = model.logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]
        # logits = model.logits_out(dec_outputs).cpu()
        # logits = logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]

        ce_loss = masked_cross_entropy(logits, Variable(target).cuda(), Variable(target_lens).cuda())
        total_loss = total_loss + ce_loss.cpu().data[0]*target_lens.float().sum()

        total_words += target_lens.sum()

        iters += 1

    print(iters)
    print(data_len)

    return total_loss / total_words.float()


def sample_outputs(model, vocab):
    model.latent_root.prune_()
    for _ in range(100):
        val1 = np.random.randint(313)
        val2 = np.random.randint(32)
        val3 = np.random.randint(38)
        val4 = np.random.randint(12)
        val5 = np.random.randint(6)
#        values = [val1, val2, 15, val4, val5]
        values = [247,12,15,val4,1]
        outputs = model.decode(values)

        print("Reconstruct: {}\n\n".format(transform(outputs, vocab.itos)))


def generate_from_seed(args, model, batches, vocab, data_len):
    """
    Generate a script from a seed tuple
    Args
        args (argparse.ArgumentParser)
        seeds (BatchIter) : BatchIter object for a file of seeds, the seed file should be in the
        same format as normal validation data
    """
    for iteration, bl in enumerate(batches):
        batch, batch_lens = bl.text
        target, target_lens = bl.target
        if args.cuda:
            batch = Variable(batch.cuda(), volatile=True)
        else:
            batch = Variable(batch, volatile=True)


        src_lens= torch.LongTensor([batch.size(1)])
        dhidden, latent_values = model(batch, src_lens, encode_only=True) #get latent encoding for seed
        model.decoder.init_feed_(Variable(torch.zeros(1, model.decoder.attn_dim)))
        _, _, dhidden, dec_outputs  = model.train(batch, 1, dhidden, latent_values, [], return_hid=True)  #decode seed

        #print("seq len {}, decode after {} steps".format(seq_len, i+1))
        # beam set current state to last word in the sequence
        beam_inp = batch[:, -1]

                # init beam initializesthe beam with the last sequence element
        outputs = model.beam_decode(beam_inp, dhidden, latent_values, args.beam_size, args.max_len_decode, init_beam=True)


        print("TRUE: {}".format(transform(batch.data.squeeze(), vocab.itos)))
        print("Reconstruct: {}\n\n".format(transform(outputs, vocab.itos)))


def reconstruct(args, model, batches, vocab):

    for iteration, bl in enumerate(batches):
        batch, batch_lens = bl.text
        target, target_lens = bl.target
        if args.cuda:
            batch = Variable(batch.cuda(), volatile=True)
        else:
            batch = Variable(batch, volatile=True)

        outputs = model(batch, batch_lens, str_out=True, beam_size=args.beam_size, max_len_decode=args.max_len_decode)

        print("TRUE: {}".format(transform(batch.data.squeeze(), vocab.itos)))
        print("Reconstruct: {}\n\n".format(transform(outputs, vocab.itos)))



def schema_constraint(cands, prev_voc, curr_verbs, min_len_decode=0, step=0, eos_idx=EOS_TOK):
    """
    Constraints to use during decoding,
    Prevents the model from producing schemas that are obviously wrong (have repeated
    predicates or the same arguments as subject and object

    Args:
        cands (Tensor [batch x vocab]) : the probabilities over the vocab for each batch/beam
        prev_voc (Tensor [batch]) : the previous output for each batch/beam
        curr_verbs (list of lists [batch x *]) : A list of lists whose kth element is a list of vocab ids of previously used
        predicates in the kth beam
        tup_idx (int) : the vocab id of the <TUP> symbol
    """
    LOW = -1e20
    K = cands.shape[0]

    for i in range(K): #for each beam
        #Replace previous vocabulary items with low probability
        beam_prev_voc = prev_voc[i]
        cands[i, beam_prev_voc] = LOW

        #Replace verbs already used with low probability
        for verb in curr_verbs[i]:
            cands[i, verb] = LOW

        if step < min_len_decode:
            cands[i, eos_idx] = LOW

    return cands

def update_verb_list(verb_list, b, tup_idx=4):
    """
    Update currently used verbs for Beam b
    verb_list is a beam_size sized list of list, with the ith list having a list of verb ids used in the ith beam
    so far
    """
    #First need to update based on prev ks
    if len(b.prev_ks) > 1:
        new_verb_list = [[]]*b.size
        for i in range(b.size):
            new_verb_list[i] = list(verb_list[b.prev_ks[-1][i]])
    else:
        new_verb_list =verb_list

    #update the actual lists
    if len(b.next_ys) == 2:
        for i, li in enumerate(new_verb_list):
            li.append(b.next_ys[-1][i])

    elif len(b.next_ys) > 2:
        for i, li in enumerate(new_verb_list):
            if b.next_ys[-2][b.prev_ks[-1][i]] == tup_idx:
                li.append(b.next_ys[-1][i])

    return new_verb_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAVAE')
    parser.add_argument('--impute_with', type=int, default=0)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--load', type=str,  default='model.pt',help='path to load the final model')
    parser.add_argument('--latent', type=str, help='A str in form of python list')
    parser.add_argument('--beam_size',  type=int, default=-1, help='Beam size')
    parser.add_argument('-perplexity',  action='store_true')
    parser.add_argument('-schema',  action='store_true')
    parser.add_argument('-nohier',  action='store_true')
    parser.add_argument('-max_len_decode', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('--n_best', type=int, default=1, help="""outputs the n_best decoded sentences""")
    parser.add_argument('--ranking',  action='store_true', help="""N cloze ranking""")
    parser.add_argument('--max_decode', type=int, default=2000, help="""max sentences to be evaluated/decoded.""")
    parser.add_argument('--num_clauses', type=int,default=5)
    parser.add_argument('--obsv_prob',  type=float, default=0, help='Beam size')
    parser.add_argument('--NYT_Noah_type',type=str,help='Noah val or test? inverse narrative cloze?')
    parser.add_argument('--mode', type=str,default="valid")

    path = os.path.dirname(os.path.realpath(__file__))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)    

    args.frame_max = 500

# ---------------  Wiki Inverse Narrative Cloze ---------------------
    args.template = 20
    path='/scratch/mehdi/'
    args.vocab = path+'/torch-event/vetm_gumbel/big_/vocab_pkl/vocab_40064_verb_max_13572.pkl'
    # args.valid_data = path+'/../torch-event/vetm_gumbel/big_/wiki_0.6M_prep/valid_0.6_TUP.txt'

    ''' NOTE: We don't use these frames for validation but the data loader needs it (we replace actual frames with __NOFRAME__'''
    args.valid_frames=path+'/torch-event/vetm_gumbel/big_/wiki_0.6M_prep/valid_0.6_frame.txt'
    args.valid_narr = path+'/torch-event/vetm_gumbel/big_/wiki_6_inverse_cloze/{}_0.6_TUP_DIST.txt'.format(args.mode)
    args.frame_vocab_address = path+'/torch-event/vetm_gumbel/big_/vocab_frame_pkl/vocab_frame_'+str(args.frame_max)+'.pkl'

    config_prefix= path+'/chain_saved_configs/'
    model_prefix= path+'/chain_saved_models/'
    config_address = '/scratch/mehdi/chain_saved_configs/chain__emb_size_300_nlayers_2_lr_0.001_batch_size_150_bidir_True_num_latent_values_500_latent_dim_500_use_pretrained_True_dropout_0.0_num_clauses_5_frame_max_500_obsv_prob_{}_template_20_model_chain_vocab_40064.pkl'.format(args.obsv_prob)
    model_address = '/scratch/mehdi/chain_saved_models/chain__emb_size_300_nlayers_2_lr_0.001_batch_size_150_bidir_True_num_latent_values_500_latent_dim_500_use_pretrained_True_dropout_0.0_num_clauses_5_frame_max_500_obsv_prob_{}_template_20_model_chain_vocab_40064.pt'.format(args.obsv_prob)
    print('prob: ',args.obsv_prob)
    print('InvData: ',args.valid_narr)
    print('config_address: ',config_address)
    print('model_address: ',model_address)    
    args.ranking=True
    with open(config_address, 'rb') as f:
        args_dict, args_info = pickle.load(f)
        args.load = model_address
        ranked_acc = generate(args)
        print('ranked_acc: ',ranked_acc)
        print('InvData: ',args.valid_narr)
        args_dict["WikiTestInvNarClz"] = ranked_acc
        print('prob: ',args.obsv_prob)
    # with open (config_address, "wb") as writer:
    #     pickle.dump((args_dict,args_info),writer)

# ---------------  Wiki Perplexity Test---------------------
    # args.template = 20
    # args.vocab = path+'/../torch-event/vetm_gumbel/big_/vocab_pkl/vocab_40064_verb_max_13572.pkl'
    # args.valid_data = path+'/../torch-event/vetm_gumbel/big_/wiki_0.6M_prep/test_0.6_TUP.txt'
    # args.perplexity=True

    # ''' NOTE: We don't use these frames for validation but the data loader needs it (we replace actual frames with __NOFRAME__'''
    # args.valid_frames=path+'/../torch-event/vetm_gumbel/big_/wiki_0.6M_prep/test_0.6_frame.txt'
    # args.valid_narr = path+'/../torch-event/vetm_gumbel/big_/wiki_6_inverse_cloze/test_0.6_TUP_DIST.txt'
    # args.frame_vocab_address = path+'/../torch-event/vetm_gumbel/big_/vocab_frame_pkl/vocab_frame_'+str(args.frame_max)+'.pkl'

    # config_prefix= path+'/chain_saved_configs/'
    # model_prefix= path+'/chain_saved_models/'
    # config_address = config_prefix+'chain__emb_size_300_nlayers_2_lr_0.001_batch_size_150_bidir_True_num_latent_values_500_latent_dim_500_use_pretrained_True_dropout_0.0_num_clauses_5_frame_max_500_obsv_prob_{}_template_20_model_chain_vocab_40064.pkl'.format(args.obsv_prob)
    # model_address = model_prefix+'chain__emb_size_300_nlayers_2_lr_0.001_batch_size_150_bidir_True_num_latent_values_500_latent_dim_500_use_pretrained_True_dropout_0.0_num_clauses_5_frame_max_500_obsv_prob_{}_template_20_model_chain_vocab_40064.pt'.format(args.obsv_prob)
    # print('prob: ',args.obsv_prob)
    # print('perplexity_data: ',args.valid_data)
    # print('config_address: ',config_address)
    # print('model_address: ',model_address)    
    # with open(config_address, 'rb') as f:
    #     args_dict, args_info = pickle.load(f)
    #     args.load = model_address
    #     ppl = generate(args)
    #     print('perplexity: ',ppl)
    #     print('perplexity_data: ',args.valid_data)
    #     args_dict["WikiTestPPL"] = ppl
    #     print('prob: ',args.obsv_prob)
    # with open (config_address, "wb") as writer:
    #     pickle.dump((args_dict,args_info),writer)

# ---------------  Noah NYT Inverse Narrative Cloze ---------------------
    # args.template = 20
    # args.vocab = path+'/../torch-event/vetm_gumbel/big_/vocab_pkl/vocab_40064_verb_max_13572.pkl'
    # args.valid_narr = path+'/../torch-event/vetm_gumbel/big_/NYT_Noah_Inverse_Cloze_Data/inverse_cloze_{}.txt'.format(args.NYT_Noah_type)

    # args.frame_vocab_address = path+'/../torch-event/vetm_gumbel/big_/vocab_frame_pkl/vocab_frame_'+str(args.frame_max)+'.pkl'

    # config_prefix= path+'/chain_saved_configs/'
    # model_prefix= path+'/chain_saved_models/'
    # config_address = config_prefix+'chain__emb_size_300_nlayers_2_lr_0.001_batch_size_150_bidir_True_num_latent_values_500_latent_dim_500_use_pretrained_True_dropout_0.0_num_clauses_5_frame_max_500_obsv_prob_{}_template_20_model_chain_vocab_40064.pkl'.format(args.obsv_prob)
    # model_address = model_prefix+'chain__emb_size_300_nlayers_2_lr_0.001_batch_size_150_bidir_True_num_latent_values_500_latent_dim_500_use_pretrained_True_dropout_0.0_num_clauses_5_frame_max_500_obsv_prob_{}_template_20_model_chain_vocab_40064.pt'.format(args.obsv_prob)
    # print('prob: ',args.obsv_prob)
    # print('config_address: ',config_address)
    # print('model_address: ',model_address)    
    # args.ranking=True
    # with open(config_address, 'rb') as f:
    #     args_dict, args_info = pickle.load(f)
    #     args.load = model_address
    #     ranked_acc = generate(args)
    #     print('{}_ranked_acc:{}'.format(args.NYT_Noah_type,ranked_acc))
    #     key='NYT_out_{}'.format(args.NYT_Noah_type)
    #     print(key)
    #     args_dict[key] = ranked_acc
    #     print('prob: ',args.obsv_prob)

    # with open (config_address, "wb") as writer:
    #     pickle.dump((args_dict,args_info),writer)





    # generate(args)








