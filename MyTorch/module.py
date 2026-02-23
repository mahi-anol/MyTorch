from typing import Dict,List,Tuple,Any,Sequence

class Parameter:
    """A trainable paramter in a neural network"""
    def __init__(self,value:Any):
        self.value=value

    @property
    def shape(self):
        """Return parameter shape if available"""
        if hasattr(self.value,"shape"):
            return self.value.shape
        return ()
    
    def __repr__(self):
        return f"Paramter (value={self.value})"
    
class Module:
    """Base class for all neural network modules"""
    def __init__(self):
        self._modules={}
        self._parameters={}
        self.training=True

    def modules(self)->Sequence["Module"]:
        """Return all sub-modules recursively (depth-first)"""
        results=[]
        for module in self._modules.values():
            results.append(module)
            results.extend(module.modules())
        return results


    def train(self):
        """Set training model for this module and all sub-modules."""
        self.training=True
        for module in self.modules():
            module.training=self.training
    
    def eval(self):
        self.training=False
        for module in self.modules():
            module.training=self.training

    
    def named_parameters(self)->Sequence[Tuple[str,Parameter]]:
        """Return all parameters with their full paths."""
        results=[]
        # Add direct parameters
        for name,param in self._parameters.items():
            results.append((name,param))
        # Add Parameter from sub module with prefix names.
        for module_name,module in self._modules.items():
            for param_name,param in module.named_parameters():
                full_name=f"{module_name}.{param_name}"
                results.append((full_name,param))
        return results

    def parameters(self)->Sequence[Parameter]:
        """Return all parameters in the module tree"""
        return [param for _, param in self.named_parameters()]
    
    def add_parameter(self,name:str,value:Any)-> Parameter:
        """Add a parameter to this module"""
        if isinstance(value,Parameter):
            param=value
        else:
            param=Parameter(value)

        self._parameters[name]=param
        return param
    
    def __setattr__(self, key:str, value:Any):
        """Automatically register modules and parameters."""
        if isinstance(value,Parameter):
            self._parameters[key]=value
        elif isinstance(value,Module):
            self._modules[key]=value
 
        super().__setattr__(key,value)


