import torch
import torch.nn as nn
from sympy.unify.core import index

from mmkd.regularization import Perturbation, Regularization, RegParameters
# from semseg.models.segformer.seg_block_UMDt import Seg as Seg_s
# from semseg.models.segformer.seg_block_select_UMD import Seg


def RUMD( x_all: list, x_all_t: list, reg_params:RegParameters):
    loss_rumd = 0.0
    loss_kl = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
    B = len(x_all)
    for i in range(B):
        for j in range(4):
            # feature_t = torch.mean(x_all_t[i][j],dim=0)
            for k in range(len(x_all[i][j])):
                # x_all[i][j][k, :, :, :] = torch.log_softmax(x_all[i][j][k, :, :, :], dim=1)
                # x_all_s[i][j][index[k], :, :, :] = torch.softmax(x_all_s[i][j][index[k], :, :, :], dim=1)
                #
                # loss_s = loss_kl(x_all[i][j][k, :, :, :], x_all_s[i][j][index[k], :, :, :]).clamp(min=0)
                # # loss_umd += loss_s
                # loss_umd = loss_umd + loss_s

                feature_s = x_all[i][j][k, :, :, :]

                feature_t = x_all_t[i][j][k, :, :, :]

                inf_feature_s = Perturbation.perturb_tensor(feature_s, reg_params.n_samples)
                inf_feature_t = Perturbation.perturb_tensor(feature_t, reg_params.n_samples)

                inf_feature_s = torch.log_softmax(inf_feature_s,dim=1)
                inf_feature_t = torch.softmax(inf_feature_t,dim=1)



                inf_loss = loss_kl(inf_feature_s, inf_feature_t).clamp(min=0)


                gradients = torch.autograd.grad(inf_loss, [inf_feature_s, inf_feature_t], create_graph=True)
                # print("inf_feature_t 的梯度值示例:", gradients[1])
                grads = [Regularization.get_batch_norm(gradients[k], loss=inf_loss,
                                                       estimation=reg_params.estimation) for k in range(2)]
                inf_scores = torch.stack(grads)
                reg_term = Regularization.get_regularization_term(inf_scores, norm=reg_params.norm,
                                                                  optim_method=reg_params.optim_method)

                loss_rumd += reg_params.lambda_ * reg_term
                # loss_rumd += inf_loss

    return loss_rumd / B

#
# model = Seg("mit_b0", num_classes=19, pretrained=True)
# model_s = Seg_s("mit_b0", num_classes=19, pretrained=True)
#
# sample = [torch.zeros(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024),
#           torch.ones(2, 3, 1024, 1024)]
#
# logits, index, ms_feat = model(sample)
# print(ms_feat[0][0].shape)
# with torch.no_grad():
#     logits_s, ms_feat_s = model_s(sample)
# print(ms_feat_s[0][0].shape)
#
# loss = UMD(index, ms_feat, ms_feat_s)
# print(0)