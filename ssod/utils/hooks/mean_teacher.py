from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n
import warnings

@HOOKS.register_module()
class MeanTeacher(Hook):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher")
        assert hasattr(model, "student")
        # only do it at initial stage
        if runner.iter == 0:
            log_every_n("Clone all parameters of student to teacher...")
            self.momentum_update(model, 0)

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        runner.log_buffer.output["ema_momentum"] = momentum
        self.momentum_update(model, momentum)

    def after_train_iter(self, runner):
        curr_step = runner.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)


@HOOKS.register_module()
class MeanSLNet(Hook):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "slnet_teacher")
        assert hasattr(model, "slnet_student")
        # only do it at initial stage
        if runner.iter == 0:
            log_every_n("Clone all parameters of student to teacher...")
            self.momentum_update(model, 0)

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        runner.log_buffer.output["sl_ema_momentum"] = momentum
        self.momentum_update(model, momentum)

    def after_train_iter(self, runner):
        curr_step = runner.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.slnet_student.named_parameters(), model.slnet_teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)


@HOOKS.register_module()
class Unlabel_weight(Hook):
    def __init__(
        self,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "ul_sl_weight")
        # only do it at initial stage
        if runner.iter == 0:
            model.ul_sl_weight = 0

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        ul_sl_weight = min(
            1, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        runner.log_buffer.output["ul_sl_weight"] = ul_sl_weight
        model.ul_sl_weight = ul_sl_weight
        # self.momentum_update(model, momentum)

    # def after_train_iter(self, runner):
    #     curr_step = runner.iter
    #     if self.decay_intervals is None:
    #         return
    #     self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
    #         self.decay_intervals, curr_step
    #     )


@HOOKS.register_module()
class Unlabel_weight_v2(Hook):
    def __init__(
        self,
        interval=1,
        warm_up=200,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "unsup_weight")
        self.cfg_unsup_weight = model.train_cfg.unsup_weight
        # only do it at initial stage
        # if runner.iter == 0:
        #     model.ul_sl_weight = 0

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        unsup_weight = self.cfg_unsup_weight * min(
            1, curr_step / (1 + self.warm_up)
        )
        # runner.log_buffer.output["unsup_weight"] = unsup_weight
        model.unsup_weight = unsup_weight


@HOOKS.register_module()
class MeanRFNet(Hook):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "rfnet_teacher")
        assert hasattr(model, "rfnet_student")
        # only do it at initial stage
        if runner.iter == 0:
            log_every_n("Clone all parameters of student to teacher...")
            self.momentum_update(model, 0)

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        runner.log_buffer.output["sl_ema_momentum"] = momentum
        self.momentum_update(model, momentum)

    def after_train_iter(self, runner):
        curr_step = runner.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.rfnet_student.named_parameters(), model.rfnet_teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)



@HOOKS.register_module()
class MeanTeacherNoDecay(Hook):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher")
        assert hasattr(model, "student")
        # only do it at initial stage
        if runner.iter == 0:
            log_every_n("Clone all parameters of student to teacher...")
            self.momentum_update(model, 0)
        # get lr imformation
        if runner.hooks[0].__class__.__name__ != 'StepLrUpdaterHook':
            warnings.warn('The lr step is not properly acquired! ! ! The ema freeze will not be applied', RuntimeWarning)
            self.lr_step = None
        else:
            self.lr_step = runner.hooks[0].step


    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        if self.lr_step and curr_step > self.lr_step[0]:
            return 
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        runner.log_buffer.output["ema_momentum"] = momentum
        self.momentum_update(model, momentum)

    def after_train_iter(self, runner):
        curr_step = runner.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
