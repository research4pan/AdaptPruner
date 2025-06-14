import torch
import torch.nn as nn
import typing
import copy
from collections import Counter

from .scheduler import linear_scheduler
from ..import function
from ... import ops, dependency
from LLMPruner.evaluator.eval import print_memory_usage, print_gpu_memory, print_model_architecture_and_parameters


class MetaPruner:
    """
        Meta Pruner for structural pruning.

        Args:
            model (nn.Module): A to-be-pruned model
            example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            importance (Callable): importance estimator. 
            ch_sparsity (float): global channel sparisty.
            ch_sparsity_dict (Dict[nn.Module, float]): layer-specific sparsity.
            iterative_steps (int): number of steps for iterative pruning.
            iterative_sparsity_scheduler (Callable): scheduler for iterative pruning.
            max_ch_sparsity (float): maximum channel sparsity.
            customized_pruners (dict): a dict containing module-pruner pairs.
        """

    def __init__(
        self,
        # Basic
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance: typing.Callable,
        layer_importance: typing.Callable,
        ch_sparsity: float = 0.5,  # channel/dim sparsity
        max_ch_sparsity: float = 1.0,
        iterative_steps: int = 1,  # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler,

        # for grouped channels.
        channel_groups: typing.Dict[nn.Module, int] = dict(),
        # for consecutive channels.
        consecutive_groups: typing.Dict[nn.Module, int] = dict(),
        # pruners for customized layers
        customized_pruners: typing.Dict[typing.Any,
                                        function.BasePruningFunc] = None,
        root_instances: typing.List = None,
    ):
        self.model = model
        self.importance = importance
        self.layer_importance = layer_importance
        self.ch_sparsity = ch_sparsity
        self.max_ch_sparsity = max_ch_sparsity

        self.channel_groups = channel_groups
        self.consecutive_groups = consecutive_groups
        self.root_instances = root_instances

        # used for relink model
        self.example_inputs = example_inputs
        self.customized_pruners = customized_pruners

        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            customized_pruners=customized_pruners,
        )

        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        # Record initial status
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                if self.DG.get_out_channels(m) is not None:
                    self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                if self.DG.get_in_channels(m) is not None:
                    self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

        # initialize ch_sparsity_dict to be uniform
        ch_sparsity_dict = {}
        for group in self.DG.get_all_groups(root_instances=self.root_instances):
            if self._check_sparsity(group):
                module = group[0][0].target.module
                ch_sparsity_dict[module] = self.ch_sparsity

        self.per_step_ch_sparsity, self.per_step_sparsity_increment = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )

        self.ch_sparsity_dict = {}
        for module in ch_sparsity_dict:
            for submodule in module.modules():
                prunable_types = tuple([ops.type2class(
                    prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                if isinstance(submodule, prunable_types):
                    self.ch_sparsity_dict[submodule] = self.per_step_ch_sparsity.copy()

        # detect group convs & group norms
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_CONV) \
                and m.groups > 1 \
                    and m.groups != m.out_channels:
                self.channel_groups[m] = m.groups
            if isinstance(m, ops.TORCH_GROUPNORM):
                self.channel_groups[m] = m.num_groups
    
    def rebuild_DG(self, model):
        # rebuild dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=self.example_inputs,
            customized_pruners=self.customized_pruners,
        )

    def get_target_sparsity(self, module, cur_step=-1):
        if cur_step == -1:
            s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[
                self.current_step]
        else:
            s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[
                cur_step]
        return min(s, self.max_ch_sparsity)

    def step(self):
        self.current_step += 1
        
        for group in self.prune_local():
            group.prune()

    def estimate_importance(self, group, ch_groups=1, consecutive_groups=1):
        return self.importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
    
    def estimate_layer_importance(self, example_prompts, lower_is_better, batch_size):
        """ Generate a dict where key is transformer layer and value is coresponding importance, normalized to [-1,1] with mean=0 """
        layer_imp_dict = {}
        layer_imp_dict_by_index = {}

        module_to_index = {layer: idx for idx, layer in enumerate(self.model.model.layers)}

        def hook_fn(module, input, output):
            input = input[0]
            output = output[0]
            assert input.shape == output.shape
            layer_imp = self.layer_importance(input, output, dim=-1, lower_is_better=lower_is_better)
            layer_imp_mean = layer_imp.mean().item()

            if module in layer_imp_dict:
                layer_imp_dict[module] += layer_imp_mean
            else:
                layer_imp_dict[module] = layer_imp_mean
            
            layer_index = module_to_index[module]
            if layer_index in layer_imp_dict_by_index:
                layer_imp_dict_by_index[layer_index] += layer_imp_mean
            else:
                layer_imp_dict_by_index[layer_index] = layer_imp_mean

        hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(lambda module, input, output: hook_fn(module, input, output))
            hooks.append(hook)
        
        with torch.no_grad():
            for mini_batch in torch.split(example_prompts, batch_size):
                _ = self.model(mini_batch)

        for hook in hooks:
            hook.remove()
        
        # expand from transformer layer to each submodule that are in root_instances
        in_use_layer_imp_dict = {}
        for transformer_layer, imp_value in layer_imp_dict.items():
            for sub_module in transformer_layer.modules():
                if (isinstance(sub_module, nn.Linear) 
                        and sub_module in self.ch_sparsity_dict.keys() 
                        and sub_module in self.root_instances):
                    in_use_layer_imp_dict[sub_module] = imp_value
        
        # normalize
        values = list(in_use_layer_imp_dict.values())
        mean_value = sum(values) / len(values)
        in_use_layer_imp_dict = {module: value - mean_value for module, value in in_use_layer_imp_dict.items()}
        max_abs_value = max(abs(value) for value in in_use_layer_imp_dict.values())
        in_use_layer_imp_dict = {module: value / max_abs_value for module, value in in_use_layer_imp_dict.items()}

        values = list(layer_imp_dict_by_index.values())
        mean_value = sum(values) / len(values)
        layer_imp_dict_by_index = {layer_idx: value - mean_value for layer_idx, value in layer_imp_dict_by_index.items()}
        max_abs_value = max(abs(value) for value in layer_imp_dict_by_index.values())
        layer_imp_dict_by_index = {layer_idx: value / max_abs_value for layer_idx, value in layer_imp_dict_by_index.items()}
        
        return in_use_layer_imp_dict, layer_imp_dict_by_index
    
    def adaptive_update_prune_distribution(self, example_prompts, lower_is_better, amplitude, batch_size):
        normalized_layer_imp_dict, layer_imp_dict_by_index = self.estimate_layer_importance(example_prompts, lower_is_better, batch_size)

        prev_step = self.current_step
        cur_step = self.current_step + 1
        for layer in self.ch_sparsity_dict.keys():
            self.ch_sparsity_dict[layer][cur_step] = max(min(
                self.ch_sparsity_dict[layer][prev_step] + 
                self.per_step_sparsity_increment - 
                amplitude * normalized_layer_imp_dict[layer], 
                0.99
            ), 0)
        return layer_imp_dict_by_index
    
    def get_least_important_layer_iex(self, example_prompts, lower_is_better):
        layer_imp_list = []

        def hook_fn(module, input, output):
            input = input[0]
            output = output[0]
            assert input.shape == output.shape
            layer_imp = self.layer_importance(input, output, dim=-1, lower_is_better=lower_is_better)
            layer_imp_mean = layer_imp.mean().item()

            layer_imp_list.append(layer_imp_mean)

        hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(lambda module, input, output: hook_fn(module, input, output))
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(example_prompts)

        for hook in hooks:
            hook.remove()
        
        # expand from transformer layer to each submodule that are in root_instances
        min_idx = layer_imp_list.index(min(layer_imp_list))
        return min_idx

    def _check_sparsity(self, group):
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None: continue
                if layer_out_ch < self.layer_init_out_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_out_ch == 1:
                    return False

            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None: continue
                if layer_in_ch < self.layer_init_in_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_in_ch == 1:
                    return False
        return True

    def get_channel_groups(self, group):
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.channel_groups:
                return self.channel_groups[module]
        return 1  # no channel grouping
    
    def get_consecutive_groups(self, group):
        if isinstance(self.consecutive_groups, int):
            return self.consecutive_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.consecutive_groups:
                return self.consecutive_groups[module]
        return 1  # no channel grouping

    def prune_local(self):
        if self.current_step > self.iterative_steps:
            return
        for group in self.DG.get_all_groups(root_instances=self.root_instances):
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
                if imp is None: continue
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )
                    
                if n_pruned <= 0:
                    continue

                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]

                if consecutive_groups > 1:
                    imp = imp.view(-1, consecutive_groups).sum(1)

                imp_argsort = torch.argsort(imp)
                
                if ch_groups > 1:
                    pruning_idxs = imp_argsort[:(n_pruned//ch_groups)]
                    group_size = current_channels//ch_groups
                    if len(pruning_idxs) > 0:
                        pruning_idxs = torch.cat(
                            [pruning_idxs+group_size*i for i in range(ch_groups)], 0)
                    else:
                        pruning_idxs = torch.tensor([])
                elif consecutive_groups > 1:
                    pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]
                    group_size = consecutive_groups
                    if len(pruning_groups) > 0:
                        pruning_idxs = torch.cat(
                            [torch.tensor([j+group_size*i for j in range(group_size)])
                            for i in pruning_groups], 0)
                    else:
                        pruning_idxs = torch.tensor([])
                else:
                    pruning_idxs = imp_argsort[:n_pruned]
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist())
                if self.DG.check_pruning_group(group):
                    yield group
