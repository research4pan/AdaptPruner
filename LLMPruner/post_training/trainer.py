import torch
import torch.nn.functional as F
import transformers
from torch.optim.lr_scheduler import LRScheduler
import math

from LLMPruner.utils.GPU_tracker import GPUMemoryTracker, DetailedMemoryProfiler

TEMPERATURE = 2.0

class LogitsTrainer(transformers.Trainer):
    def __init__(self, *args, teacher_model=None, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.total_tokens = 0
        self.alpha=alpha
        
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        self.total_tokens += inputs['input_ids'].numel()

        if self.teacher_model is None:
            student_outputs = model(**inputs)
            return (student_outputs.loss, student_outputs) if return_outputs else student_outputs.loss

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        student_outputs = model(**inputs)

        distillation_loss = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            student_outputs.loss,
            inputs['labels'],
            next(model.parameters()).dtype
        )

        return (distillation_loss, student_outputs) if return_outputs else distillation_loss

    def distillation_loss(self, student_logits, teacher_logits, original_loss, labels, dtype):
        labels_mask = (labels != -100).float()  # [batch_size, seq_len]

        student_logits = student_logits / TEMPERATURE
        teacher_logits = teacher_logits / TEMPERATURE

        loss_kd = F.kl_div(
            F.log_softmax(student_logits, dim=-1).to(dtype),
            F.softmax(teacher_logits, dim=-1).to(dtype),
            reduction='none',
            log_target=False
        )  # [batch_size, seq_len, vocab_size]
        
        loss_kd = loss_kd.sum(dim=-1)  # [batch_size, seq_len]
        loss_kd = (loss_kd * labels_mask).sum() / labels_mask.sum()
        
        loss_kd = loss_kd * (TEMPERATURE ** 2)
        
        if self.alpha == 1.0:
            return loss_kd

        return self.alpha * loss_kd + (1 - self.alpha) * original_loss

class CustomCosineScheduler(LRScheduler):
    def __init__(
        self, 
        optimizer, 
        total_training_steps: int,
        base_training_steps: int,
        total_warmup_steps: int,
        min_lr: float = 1e-8,
        last_epoch: int = -1,
    ):
        self.total_training_steps = total_training_steps
        self.base_training_steps = base_training_steps
        self.total_warmup_steps = total_warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cur_step = self.base_training_steps + self.last_epoch
        if cur_step < self.total_warmup_steps:
            warmup_progress = float(cur_step) / float(max(1, self.total_warmup_steps))
            return [base_lr * warmup_progress for base_lr in self.base_lrs]
        else:
            progress = min(1.0, float(cur_step - self.total_warmup_steps) / float(max(1, self.total_training_steps - self.total_warmup_steps)))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

class CustomWSDScheduler(LRScheduler):
    def __init__(
        self, 
        optimizer, 
        total_training_steps: int,
        base_training_steps: int,
        total_warmup_steps: int,
        min_lr: float = 1e-8,
        last_epoch: int = -1,
    ):
        self.total_training_steps = total_training_steps
        self.base_training_steps = base_training_steps
        self.total_warmup_steps = total_warmup_steps
        self.min_lr = min_lr
        
        # Calculate the step where decay phase starts (90% of total steps)
        self.decay_start_step = int(0.9 * total_training_steps)
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cur_step = self.base_training_steps + self.last_epoch
        
        if cur_step < self.total_warmup_steps:
            warmup_progress = float(cur_step) / float(max(1, self.total_warmup_steps))
            return [base_lr * warmup_progress for base_lr in self.base_lrs]
        
        # Decay phase (last 10% of training)
        elif cur_step >= self.decay_start_step:
            decay_steps = self.total_training_steps - self.decay_start_step
            decay_progress = float(cur_step - self.decay_start_step) / float(max(1, decay_steps))
            return [
                max(base_lr - (base_lr - self.min_lr) * decay_progress, self.min_lr)
                for base_lr in self.base_lrs
            ]
        
        else:
            return [base_lr for base_lr in self.base_lrs]