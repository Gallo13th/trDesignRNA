import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def structure_loss_with_pdb_template(pred, template, mask):
    losses = {
        k: 0 for k in ['P','C3\'','C1\'','N1'] # C4
    }
    pred = pred['distance']
    template = template['distance']
    for k in pred:
        if k in template:
            pred_ = pred[k][mask[:,0],:,:][:,mask[:,0],:]
            # print(pred_.shape)
            template_ = template[k][mask[:,1],:,:][:,mask[:,1],:]
            # print(pred[k].shape,pred_.shape,template_.shape)
            loss = F.cross_entropy(pred_, template_, reduction='mean')
            losses[k] = loss
    return losses

def hallucinate_loss(pred, background, mask, valid=False):
    losses = {
        'template':{
        'distance':{k: 0 for k in ['P','C3\'','C1\'','N1']},
        'contact': 0
        },
        'impainting':{
        'distance':{k: 0 for k in ['P','C3\'','C1\'','N1']},
        'contact': 0
        }
    }
    device = pred['distance']['P'].device
    ssmask = mask
    klmask_template = torch.einsum('i,j->ij',ssmask,ssmask)
    klmask_impainting = 1 - klmask_template
    if valid:
        klmask_impainting = torch.ones_like(klmask_template)
        klmask_template = torch.ones_like(klmask_template)
    for l in pred:
        if isinstance(pred[l], dict):
            for k in pred[l]:
                p = pred[l][k]
                q = background[l][k].to(device).detach()
                kl = p * torch.log((p+1e-6) / (q+1e-6))
                kl_template = (kl * klmask_template.unsqueeze(-1)).sum() / klmask_template.unsqueeze(-1).sum()
                kl_impainting =  (kl * klmask_impainting.unsqueeze(-1)).sum() / klmask_impainting.unsqueeze(-1).sum()
                losses['template'][l][k] = kl_template
                losses['impainting'][l][k] = kl_impainting
                loss = kl_template - kl_impainting
                assert loss.isnan().any() == False,(p.max(),p.min(),q.max(),q.min())
        else:
            p = pred[l]
            q = background[l].to(device).detach()
            # print(p)
            kl = p * torch.log((p+1e-6) / (q+1e-6))
            kl_template = (kl * klmask_template).sum() / klmask_template.sum()
            kl_impainting =  (kl * klmask_impainting).sum() / klmask_impainting.sum()
            losses['template'][l] = kl_template
            losses['impainting'][l] = kl_impainting
            loss = kl_template - kl_impainting
            assert loss.isnan().any() == False,(p.max(),p.min(),q.max(),q.min())
    return losses

if __name__ == '__main__':
    pred = {
        'distance': {
            'P': torch.rand(10,10,10),
            'C3\'': torch.rand(10,10,10),
            'C1\'': torch.rand(10,10,10),
            'C4': torch.rand(10,10,10),
            'N1': torch.rand(10,10,10)
        }
    }
    template = {
        'distance': {
            'P': torch.rand(10,10,10),
            'C3\'': torch.rand(10,10,10),
            'C1\'': torch.rand(10,10,10),
            'C4': torch.rand(10,10,10),
            'N1': torch.rand(10,10,10)
        }
    }
    mask = np.array(
        [[0,0],
         [1,1],
         [2,2],
         [3,3],
         [4,4],
         [5,5],
         [6,6],
         [7,8],
         [8,9]]
    )
    print(structure_loss_with_pdb_template(pred,template,mask))
    mask = np.array(
        [0,1,2,6,7,8,9]
    )
    print(hallucinate_loss(pred,template,mask))
            