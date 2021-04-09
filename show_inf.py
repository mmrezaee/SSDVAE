import torch
import numpy as np

def show_inference(model,batch,vocab,vocab2,f_vals,f_ref,args):
    latent_gumbels=model.latent_gumbels
    frame_to_vocab = model.frame_to_vocab
    scores = model.scores[0,0,:,:].data.cpu().squeeze()
    _,scores = torch.sort(scores,-1,descending=True)
    scores = scores[:,:15]
    template_embedding = model.template_to_frame.weight.data.cpu()
    _,template_sort = torch.sort(template_embedding.t(),-1,descending=True)
    template_sort=template_sort[:,:15]
    template_dict={}
    for k in range(template_sort.size(0)):
        template_meaning = [vocab2.itos[int(v.numpy())] for v in template_sort[k,:]]
        template_dict[k] = template_meaning

    latent_gumbels = model.latent_gumbels
    frames_to_frames = model.frames_to_frames
    frame_to_frame = frames_to_frames[0,:,:].data.cpu()
    next_frames, next_frames_indicies = torch.sort(frame_to_frame,-1,descending=True)
    next_frames = next_frames[:,:15]
    next_frames_indicies = next_frames_indicies[:,:15]

    word_to_frame={}
    real_sentence = batch[0,:].data.cpu()
    real_sentence = [vocab.itos[int(v.numpy())] for v in real_sentence]
    for k in range(len(real_sentence)):
        if k%1==0:
            word_to_frame[real_sentence[k]] = [vocab2.itos[int(v.numpy())] for v in scores[k,:]]

    beta = frame_to_vocab[0,:,:].data.cpu()
    frames_infer = latent_gumbels[0,:,:].data.cpu()
    frames_infer_idx = torch.argmax(frames_infer, -1)
    real_f_vals = f_vals[0,:].data.cpu()
    ref_f = f_ref[0,:].data

    beta_sort, beta_sort_indicies = torch.sort(beta, -1,descending=True)
    beta_sort = beta_sort[: ,:15]
    beta_sort_indicies = beta_sort_indicies[: ,:15]
    topics_dict={}
    next_frames_dict={}
    topics_dict['ref_frames']=[]
    topics_dict['fval_frames']=[]
    topics_dict['infered_frames']=[]
    for k in range(args.num_clauses):
        real_frame=vocab2.itos[real_f_vals[k]]
        which_frame=vocab2.itos[frames_infer_idx[k]]
        args_meaning=[vocab.itos[item.cpu().numpy()] for item in beta_sort_indicies[k]]
        next_frame_meaning = [vocab2.itos[item.cpu().numpy()] for item in next_frames_indicies[k]]
        ref_frames_meaning = [vocab2.itos[item.cpu().numpy()] for item in ref_f]
        topics_dict[which_frame] = args_meaning
        next_frames_dict[which_frame] = next_frame_meaning
        topics_dict['infered_frames']+=[which_frame]
        topics_dict['fval_frames']+=[real_frame]
        topics_dict['ref_frames'] =ref_frames_meaning
    topics_dict['ref_frames'] += ["-"]*(15-len(topics_dict['ref_frames']))
    topics_dict['infered_frames'] += ["-"]*(15-len(topics_dict['fval_frames']))
    topics_dict['fval_frames'] += ["-"]*(15-len(topics_dict['fval_frames']))
    return topics_dict,real_sentence,next_frames_dict,word_to_frame,template_dict
