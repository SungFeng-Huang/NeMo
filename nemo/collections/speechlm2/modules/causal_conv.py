import torch
import torch.nn as nn

class FixedLengthCausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d that maintains proper causal behavior
    Supports both regular convolution (stride=1) and downsampling (stride=2)
    This is specifically designed to replace Conv1d layers in ResNet blocks and Downsample1D blocks
    Includes cache mechanism for streaming inference
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,  # Original padding value
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        # For causal convolution, we need to pad the input to maintain output length
        # Original: Conv1d(kernel_size, padding=1) -> output_length = input_length (stride=1) or input_length // 2 (stride=2)
        # Causal: We need to pad input by kernel_size-1 on the left to maintain causal behavior
        
        # Cache mechanism setup
        self.cache_drop_size = None
        
        # Calculate padding for causal behavior
        self._left_padding = dilation * (kernel_size - 1)
        self._right_padding = 0  # For stride > 1, we need right padding?
        
        # Cache length is determined by left padding
        self._max_cache_len = self._left_padding
        
        # Call parent with padding=0 since we handle padding manually
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
    
    def update_cache(self, x, cache=None):
        """
        Update cache for streaming inference
        Handles both stride=1 and stride=2 (downsampling) cases
        """
        if cache is None:
            # No cache: pad the input normally, no cache mode
            new_x = torch.nn.functional.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = None
        else:
            # Cache is not None: cache mode
            assert cache.shape[-1] >= self._left_padding, f"Cache length must be greater than or equal to left padding: {cache.shape[-1]} >= {self._left_padding}"

            # With cache: concatenate cache with new input
            new_x = torch.nn.functional.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            
            # Handle cache drop for stride > 1 (matching original CausalConv1D logic)
            if self.cache_drop_size is not None and self.cache_drop_size > 0:
                next_cache = new_x[:, :, :-self.cache_drop_size]
            else:
                next_cache = new_x
            
            # Maintain cache size, not caching the whole history
            next_cache = next_cache[:, :, -cache.size(-1):]
        
        return new_x, next_cache
    
    def forward(self, x, cache=None):
        """
        Forward pass with optional cache support
        Returns (output, cache) if cache is provided, otherwise just output
        """
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        
        if cache is None:
            return x
        else:
            return x, cache

class CausalConvTranspose1D(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self._upsample_factor = stride
        self._padding = kernel_size - 1

    def forward(self, x):
        n = x.shape[-1]

        x = torch.nn.functional.pad(x, (self._padding, 0))
        out = super().forward(x)
        out = out[..., :(n * self._upsample_factor)]

        return out


class CausalConvConverter:
    """Convert standard Conv1d layers to CausalConv1D when loading the model"""
    
    def __init__(self, model, causal_config):
        self.model = model
        self.causal_config = causal_config
        self.original_modules = {}
        self.converted_modules = {}
    
    def convert_model(self):
        """Convert all Conv1d layers in the model to CausalConv1D"""
        for name, module in self.model.named_modules():
            if self._should_convert_module(name, module):
                self._convert_module(name, module)
    
    def _should_convert_module(self, name, module):
        """Determine whether a module should be converted"""
        # Skip if already converted
        if name in self.converted_modules:
            return False
        
        # Check if it's a Conv1d or ConvTranspose1d module
        if not isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            return False
        
        # Check if module is in affected modules list
        affected_modules = self.causal_config.get('affected_modules', ['decoder'])
        for affected in affected_modules:
            if affected in name:
                return True
        
        return False
    
    def _convert_module(self, name, module):
        """Convert a single Conv1d or ConvTranspose1d module to CausalConv1D or CausalConvTranspose1D"""
        # Save the original module
        self.original_modules[name] = module
        
        if isinstance(module, nn.ConvTranspose1d):
            # Handle ConvTranspose1d conversion
            # This is the typical case for Upsample1D: ConvTranspose1d(4, 2, 1)
            causal_module = CausalConvTranspose1D(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0],
                output_padding=module.output_padding[0] if hasattr(module, 'output_padding') else 0,
                groups=module.groups,
                bias=module.bias is not None,
                dilation=module.dilation[0] if hasattr(module, 'dilation') else 1,
                padding_mode=module.padding_mode if hasattr(module, 'padding_mode') else 'zeros'
            )
        else:
            # Handle Conv1d conversion - use FixedLengthCausalConv1D for all cases
            # It now supports cache mechanism, dilation, and stride > 1
            causal_module = FixedLengthCausalConv1D(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0],  # Pass original padding for reference
                dilation=module.dilation[0] if hasattr(module, 'dilation') else 1,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode if hasattr(module, 'padding_mode') else 'zeros'
            )
        
        # Copy weights
        with torch.no_grad():
            causal_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                causal_module.bias.data = module.bias.data.clone()
        
        # Replace the module in the model
        self._replace_module_in_model(name, causal_module)
        
        # Mark as converted
        self.converted_modules[name] = causal_module
    
    def _replace_module_in_model(self, name, new_module):
        """Replace a module in the model hierarchy"""
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent_module = self.model.get_submodule(parent_name)
            setattr(parent_module, child_name, new_module)
        else:
            setattr(self.model, child_name, new_module)
    
    def restore_original_modules(self):
        """Restore original modules (if needed)"""
        for name, original_module in self.original_modules.items():
            self._replace_module_in_model(name, original_module)
        
        # Clear converted modules tracking
        self.converted_modules.clear()
    
    def get_converted_modules(self):
        """Get list of converted module names"""
        return list(self.converted_modules.keys())
    
    def is_converted(self, name):
        """Check if a module has been converted"""
        return name in self.converted_modules


class CausalConfigManager:
    """Manager for causal convolution configuration"""
    
    def __init__(self):
        self.config = {
            'causal_mode': False,
            'affected_modules': ['decoder', 'encoder'],
            'conversion_strategy': 'converter'
        }
        self.converter = None
    
    def update_config(self, new_config):
        """Update configuration"""
        self.config.update(new_config)
    
    def apply_to_model(self, model):
        """Apply configuration to model using converter strategy"""
        if self.converter is None:
            self.converter = CausalConvConverter(model, self.config)
        
        # Convert standard Conv1d to CausalConv1D if causal mode is enabled
        if self.config['causal_mode']:
            self.converter.convert_model()
    
    def cleanup(self):
        """Clean up and restore original modules if needed"""
        if self.converter is not None and not self.config['causal_mode']:
            self.converter.restore_original_modules()
            