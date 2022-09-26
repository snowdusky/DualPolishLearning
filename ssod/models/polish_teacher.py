import torch
from mmcv.runner import auto_fp16
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid
from torch import gt, nn
from torch.nn.modules.utils import _pair
from mmdet.core import BboxOverlaps2D, build_bbox_coder
from mmdet.models import build_loss

class Merger(nn.Module):
    def __init__(self, in_dim=256, mid_dim=256, last_dim=1024, vote_num=7):
        super(Merger, self).__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.cls_input = mid_dim * 7 * 7
        self.vote_num = vote_num

        self.feat_conv = nn.Conv2d(self.in_dim * vote_num, self.mid_dim, 1, stride=1, padding=0)
        self.feat_fc1 = nn.Linear(self.cls_input, last_dim)
        self.feat_fc2 = nn.Linear(last_dim, last_dim)
        self.relu = nn.ReLU()

    @auto_fp16()
    def forward(self, feat_x):
        # b, n, c, w, h
        b, n, c, w, h = feat_x.shape
        assert n == self.vote_num
        feat_x = self.relu(self.feat_conv(feat_x.reshape(b, n*c, w, h))).reshape(b, self.cls_input)
        feat_x = self.relu(self.feat_fc2(self.relu(self.feat_fc1(feat_x))))
        return feat_x

class PolishNet(nn.Module):
    def __init__(self, in_dim=256, mid_dim=256, last_dim=1024, num_classes=20, vote_num=7):
        super(PolishNet, self).__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.last_dim = last_dim
        self.num_classes = num_classes
        self.reg_merger = Merger(in_dim=in_dim, mid_dim=mid_dim, last_dim=last_dim, vote_num=vote_num)
        # self.cls_merger = Merger(in_dim=in_dim, mid_dim=mid_dim, last_dim=last_dim, vote_num=vote_num)
        self.cls_merger = nn.Sequential(
            nn.Linear(256 * 7 * 7, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, last_dim),
        )
        self.fp16_enabled = False
        
        self.reg = nn.Linear(last_dim, 4)
        self.cls = nn.Linear(last_dim, self.num_classes + 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        bbox_coder_cfg = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0])
        self.bbox_coder = build_bbox_coder(bbox_coder_cfg)
        self.giou_loss = build_loss(dict(type='GIoULoss', loss_weight=1.0))
        self.ce_loss = nn.CrossEntropyLoss()

    @auto_fp16()
    def forward(self, feat_x):
        # b, n, c, w, h
        reg_x = self.reg_merger(feat_x)
        # cls_x = self.cls_merger(feat_x)
        cls_x = self.cls_merger(feat_x[:, 0, ...].reshape(feat_x.shape[0], 256 * 7 * 7))

        deltas = self.reg(reg_x)
        classes = self.cls(cls_x)
        return deltas, classes

    @auto_fp16()
    def forward_reg(self, feat_x):
        # b, n, c, w, h
        reg_x = self.reg_merger(feat_x)
        deltas = self.reg(reg_x)
        return deltas

    @auto_fp16()
    def forward_cls(self, feat_x):
        # b, n, c, w, h
        classes = self.cls(self.cls_merger(feat_x[:, 0, ...].reshape(feat_x.shape[0], 256 * 7 * 7)))
        # classes = self.cls(self.cls_merger(feat_x))
        return classes

    def inference_reg(self, feat_x):
        deltas = self.forward_reg(feat_x)
        return deltas

    def inference_cls(self, feat_x):
        classes = self.forward_cls(feat_x)
        return self.softmax(classes)
    
    @force_fp32(apply_to=["jittered", "voted_deltas", "targets"])
    def reg_loss(self, jittered, voted_deltas, targets, num_jittered_per_img, img_metas):
        voted_deltas_list = voted_deltas.split(num_jittered_per_img, 0)
        voted_bboxes_list = [
            self.bbox_coder.decode(jittered[i], voted_deltas_list[i], max_shape=img_metas[i]['img_shape']) \
            for i in range(len(num_jittered_per_img))
        ]
        voted_bboxes = torch.cat(voted_bboxes_list)
        targets, jittered = torch.cat(targets), torch.cat(jittered)
        loss = self.giou_loss(voted_bboxes, targets)
        # delta_targets = self.bbox_coder.encode(jittered, targets)
        # loss += (voted_deltas - delta_targets).abs().mean()
        return loss, voted_bboxes_list

    @force_fp32(apply_to=["pred_scores"])
    def cls_loss(self, pred_scores, label):
        loss = self.ce_loss(pred_scores, label)
        return loss

@DETECTORS.register_module()
class PolishTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(PolishTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
            self.rf_clslossw = self.train_cfg.rf_clslossw
            self.rf_reglossw = self.train_cfg.rf_reglossw
            self.rf_vote_frac = self.train_cfg.rf_vote_frac
            self.rf_cls_thr = self.train_cfg.rf_cls_thr
            self.rf_pos_iou_thr = self.train_cfg.rf_pos_iou_thr

        self.num_classes = self.student.roi_head.bbox_head.num_classes
        self.rfnet_teacher = PolishNet(num_classes=self.num_classes, vote_num=6 + 1)
        self.rfnet_student = PolishNet(num_classes=self.num_classes, vote_num=6 + 1)
        self.iou_calculator = BboxOverlaps2D()
        for name, pa in self.rfnet_teacher.named_parameters():
            pa.requires_grad = False

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            sup_gt_num = torch.Tensor([sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)]).to(gt_bboxes[0].device)
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss.update({'sup_gt_num': sup_gt_num})
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
            # sl part
            rf_loss = self.train_rfnet(data_groups["sup"]['img'], gt_bboxes, data_groups["sup"]['gt_labels'], data_groups["sup"]["img_metas"])
            loss.update(**rf_loss)
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
            loss.update({'unsup_weight': sup_gt_num.new_full([1], self.unsup_weight)})

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def train_rfnet(self, imgs, gt_bboxes, gt_labels, img_metas):
        times = 6 + 1
        with torch.no_grad():
            feat = self.teacher.extract_feat(imgs)
        logs = self.train_rfnet_reg(times, feat, gt_bboxes, img_metas)
        logs.update(self.train_rfnet_cls(times, feat, gt_bboxes, gt_labels, img_metas))
        # torch.cuda.empty_cache()
        return logs

    def train_rfnet_reg(self, times, feat, gt_bboxes, img_metas):
        jittered = self.aug_box(gt_bboxes, self.train_cfg.rf_reg_sample_num, self.train_cfg.rf_reg_sample_scale)
        jittered = [bboxes.reshape(-1, bboxes.shape[-1]) for bboxes in jittered]
        # with torch.no_grad():
        #     proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, img_metas)
        # jittered = [torch.cat([jittered[i], proposal_list[i][:, :4], gt_bboxes[i]]) for i in range(len(gt_bboxes))]
        jittered = [torch.cat([jittered[i], gt_bboxes[i]]) for i in range(len(gt_bboxes))]
        # 筛掉 y2 < y1 和 x2 < x1 的
        jittered = [ji[(ji[:, 3] > ji[:, 1]) & (ji[:, 2] > ji[:, 0])] for ji in jittered]
        # get labels
        targets, iou_before, label = [], [], []
        for i in range(len(jittered)):
            overlaps = self.iou_calculator(gt_bboxes[i], jittered[i])
            # keep =(overlaps.max(0).values >= self.rf_pos_iou_thr)
            # jittered[i] = jittered[i][keep]
            # overlaps = self.iou_calculator(gt_bboxes[i], jittered[i])
            label.append(overlaps.max(0).indices)
            targets.append(gt_bboxes[i][label[i]])
            iou_before.append(overlaps.max(0).values.mean())
        iou_before = sum(iou_before) / len(iou_before)
        num_jittered_per_img = tuple(len(p) for p in jittered)
        voted_proposal_list = self.vote_box(jittered, self.rf_vote_frac)
        voted_proposal_list = [torch.cat([jittered[i], voted_proposal_list[i]], 0) for i in range(len(jittered))]

        # get voted jittered features
        rois = bbox2roi(voted_proposal_list)
        with torch.no_grad():
            bbox_feats = self.teacher.roi_head.bbox_roi_extractor(feat[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
        feat_input = bbox_feats.reshape(times, -1, bbox_feats.shape[1], bbox_feats.shape[2], bbox_feats.shape[3]).transpose(0, 1)

        # get preds
        voted_deltas = self.rfnet_student.forward_reg(feat_input)
        # make loss
        loss, voted_bboxes_list = self.rfnet_student.reg_loss(jittered, voted_deltas, targets, num_jittered_per_img, img_metas)

        # cal mean iou improvement
        iou_after = []
        for i in range(len(num_jittered_per_img)):
            overlaps = self.iou_calculator(gt_bboxes[i], voted_bboxes_list[i])
            iou_after.append(overlaps.max(0).values.mean())
        iou_after = sum(iou_after) / len(iou_after)

        return {'rf_regloss': loss * self.rf_reglossw, 'iou_before': iou_before, 'iou_after': iou_after}

    def train_rfnet_cls(self, times, feat, gt_bboxes, gt_labels, img_metas):
        jittered = self.aug_box(gt_bboxes, self.train_cfg.rf_cls_sample_num[0], self.train_cfg.rf_cls_sample_scale[0])
        jittered = [bboxes.reshape(-1, bboxes.shape[-1]) for bboxes in jittered]
        jittered_mid = self.aug_box(gt_bboxes, self.train_cfg.rf_cls_sample_num[1], self.train_cfg.rf_cls_sample_scale[1])
        jittered_mid = [bboxes.reshape(-1, bboxes.shape[-1]) for bboxes in jittered_mid]
        # 筛掉 y2 < y1 和 x2 < x1 的
        # torch.cuda.empty_cache()
        with torch.no_grad():
            proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, img_metas)
        jittered = [torch.cat([jittered[i], jittered_mid[i], proposal_list[i][:, :4], gt_bboxes[i]]) for i in range(len(gt_bboxes))]
        jittered = [ji[(ji[:, 3] > ji[:, 1]) & (ji[:, 2] > ji[:, 0])] for ji in jittered]
        # get labels
        label = []
        for i in range(len(jittered)):
            overlaps = self.iou_calculator(gt_bboxes[i], jittered[i])
            fg = (overlaps.max(0).values >= self.rf_pos_iou_thr)
            tmp_l = torch.full([jittered[i].shape[0]], self.num_classes).long().to(jittered[i].device)
            tmp_l[fg] = gt_labels[i][overlaps.max(0).indices][fg]
            label.append(tmp_l)
        label = torch.cat(label)

        voted_proposal_list = self.vote_box(jittered, self.rf_vote_frac)
        voted_proposal_list = [torch.cat([jittered[i], voted_proposal_list[i]], 0) for i in range(len(jittered))]

        # get voted jittered features
        rois = bbox2roi(voted_proposal_list)
        with torch.no_grad():
            bbox_feats = self.teacher.roi_head.bbox_roi_extractor(feat[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
        feat_input = bbox_feats.reshape(times, -1, bbox_feats.shape[1], bbox_feats.shape[2], bbox_feats.shape[3]).transpose(0, 1)

        # get preds
        jittered = torch.cat(jittered)
        pred_scores = self.rfnet_student.forward_cls(feat_input)
        # make loss
        loss = self.rfnet_student.cls_loss(pred_scores, label)
        _, pred_label = pred_scores.topk(1, dim=1)
        # cal acc and recall
        train_acc = torch.eq(label, pred_label[:, 0]).sum() / torch.ones_like(label).sum()
        isbg = (label == self.num_classes)
        train_pos_recall = torch.eq(label, pred_label[:, 0])[~isbg].sum() / (~isbg).sum()
        train_neg_recall = torch.eq(label, pred_label[:, 0])[isbg].sum() / isbg.sum()
        return {'rf_clsloss': loss * self.rf_clslossw, 'train_acc': train_acc, 'train_pos_recall': train_pos_recall, 'train_neg_recall': train_neg_recall}


    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        reg_pseudo_bboxes = self._transform_bbox(
            teacher_info["reg_det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        cls_pseudo_bboxes = self._transform_bbox(
            teacher_info["cls_det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )

        # pseudo_labels = teacher_info["det_labels"]
        loss = {}
        loss.update({'keep_ratio': teacher_info['keep_ratio'], 'keep_num': teacher_info['keep_num'], 'pre_num': teacher_info['pre_num'], })
        # TODO: rpn用哪个框？
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            cls_pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                cls_pseudo_bboxes,
                teacher_info["cls_det_labels"],
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        )
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                reg_pseudo_bboxes,
                teacher_info["reg_det_labels"],
                student_info=student_info,
            )
        )
        return loss

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            rpn_gt_num = torch.Tensor([sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)]).to(gt_bboxes[0].device)
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=proposal_cfg
            )
            # log_image_with_boxes(
            #     "rpn",
            #     student_info["img"][0],
            #     pseudo_bboxes[0][:, :4],
            #     bbox_tag="rpn_pseudo_label",
            #     scores=pseudo_bboxes[0][:, 4],
            #     interval=500,
            #     img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            # )
            losses.update({"rpn_gt_num": rpn_gt_num})
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        rcnn_cls_gt_num = torch.Tensor([sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)]).to(gt_bboxes[0].device)
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            # loss weight
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        loss["rcnn_cls_gt_num"] = rcnn_cls_gt_num
        # if len(gt_bboxes[0]) > 0:
        #     log_image_with_boxes(
        #         "rcnn_cls",
        #         student_info["img"][0],
        #         gt_bboxes[0],
        #         bbox_tag="pseudo_label",
        #         labels=gt_labels[0],
        #         class_names=self.CLASSES,
        #         interval=500,
        #         img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #     )
        return loss

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            thr=0,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        rcnn_reg_gt_num = torch.Tensor([sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)]).to(gt_bboxes[0].device)
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        # if len(gt_bboxes[0]) > 0:
        #     log_image_with_boxes(
        #         "rcnn_reg",
        #         student_info["img"][0],
        #         gt_bboxes[0],
        #         bbox_tag="pseudo_label",
        #         labels=gt_labels[0],
        #         class_names=self.CLASSES,
        #         interval=500,
        #         img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #     )
        return {"loss_bbox": loss_bbox, "rcnn_reg_gt_num": rcnn_reg_gt_num}

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        with torch.no_grad():
            proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
                feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
            )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        thr = self.train_cfg.pseudo_label_initial_score_thr
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        with torch.no_grad():
            img_shapes = [metas['img_shape'] for metas in img_metas]
            reg_det_bboxes, keep_list, new_label_list = self.polish(feat, proposal_list, proposal_label_list, img_shapes)
        reg_det_labels = proposal_label_list
        cls_det_bboxes = [proposal_list[i][keep_list[i]] for i in range(len(proposal_list))]
        cls_det_labels = [new_label_list[i][keep_list[i]] for i in range(len(proposal_list))]

        teacher_info["reg_det_bboxes"] = reg_det_bboxes
        teacher_info["reg_det_labels"] = reg_det_labels
        teacher_info["cls_det_bboxes"] = cls_det_bboxes
        teacher_info["cls_det_labels"] = cls_det_labels
        # cal cls keep num
        teacher_info['pre_num'] = sum([torch.Tensor([bbox.shape[0]]).to(bbox.device) for bbox in proposal_list])
        teacher_info['keep_num'] = sum([torch.Tensor([bbox.shape[0]]).to(bbox.device) for bbox in cls_det_bboxes])
        teacher_info['keep_ratio'] = teacher_info['keep_num'] / (teacher_info['pre_num'] + 1e-8)

        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def polish(self, feat, proposal_list, proposal_label_list, img_shapes):
        times = 6 + 1
        num_proposals_per_img = tuple(len(p) for p in proposal_list)
        voted_proposal_list = self.vote_box(proposal_list, self.rf_vote_frac)
        # get voted jittered features
        voted_proposal_list = [torch.cat([proposal_list[i], voted_proposal_list[i]], 0) for i in range(len(proposal_list))]
        rois = bbox2roi(voted_proposal_list)
        bbox_feats = self.teacher.roi_head.bbox_roi_extractor(feat[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
        voted_bboxes_list = self.second_reg(times, bbox_feats, proposal_list, num_proposals_per_img, img_shapes)
        keep_list, new_label_list = self.second_cls(times, bbox_feats, proposal_list, proposal_label_list, num_proposals_per_img)
        return voted_bboxes_list, keep_list, new_label_list

    def second_reg(self, times, bbox_feats, proposal_list, num_proposals_per_img, img_shapes):
        feat_input = bbox_feats.reshape(times, -1, bbox_feats.shape[1], bbox_feats.shape[2], bbox_feats.shape[3]).transpose(0, 1)
        # get preds
        voted_deltas_list = self.rfnet_teacher.inference_reg(feat_input).split(num_proposals_per_img)
        voted_bboxes_list = []
        for i in range(len(proposal_list)):
            voted_bboxes_list.append(self.rfnet_teacher.bbox_coder.decode(proposal_list[i], voted_deltas_list[i], max_shape=img_shapes[i]))

        return voted_bboxes_list

    def second_cls(self, times, bbox_feats, proposal_list, proposal_label_list, num_proposals_per_img):
        # reshape the features part
        feat_input = bbox_feats.reshape(times, -1, bbox_feats.shape[1], bbox_feats.shape[2], bbox_feats.shape[3]).transpose(0, 1)
        # get preds
        pred = self.rfnet_teacher.inference_cls(feat_input)

        scores = torch.cat(proposal_list)[:, 4:]
        fg_value, fg_label = pred[:, :-1].topk(1, dim=1)
        new_labels = torch.cat(proposal_label_list)
        cls_change = (fg_value >= self.rf_cls_thr) & (scores < 0.9)
        new_labels[:, None][cls_change] = fg_label[cls_change]
        keep = (fg_value[:, 0] >= self.rf_cls_thr) | (scores[:, 0] > 0.9)

        return keep.split(num_proposals_per_img), new_labels.split(num_proposals_per_img)

    @staticmethod
    def vote_box(boxes, frac=0.06):
        # TODO: 是否要加上原bbox
        def _jit4_single(box):
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            aug_scale = box_scale * frac  # [n,4]
            dirt = torch.ones(4, 4).to(box.device)
            dirt[:2, 1::2] *= -1
            dirt[::2, ::2] *= -1
            dirt2 = torch.Tensor([[-1, -1, 1, 1]]).to(box.device)
            dirt3 = torch.Tensor([[-1, -1, 1, 1]]).to(box.device) * 2
            dirt = torch.cat([dirt, dirt2, dirt3], 0)
            offset = dirt[:, None, ...] * aug_scale[None, ...]
            new_box = box.clone()[None, ...].expand(offset.shape[0], box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            ).reshape(-1, box.shape[1])
        return [_jit4_single(box) for box in boxes]

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
