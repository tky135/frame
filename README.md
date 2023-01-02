# frame
1. Inherit dataset class from task class. 

1. (Optional) Define class attribute `data_path`, if not defined, default to data root + name of dataset class. 

1. Define `list_data`:

   - Input: 

     - `self.path` - the dataset path

   - Output:

     - `return X1, X2, ..., Xn, Y1, Y2, ..., Ym`, lists of inputs and outputs

       > Typically `return X, Y`

   - Inout:

     - `self.dict` - universal recording (across experiments)

       > Typically used for mapping between class name and class index

   - Effects:

     List all the data in `self.path`. Do any preprocessing on raw data. This function will be called in `__init__` iff do split. The returned data type should be reasonable to store in csv files. 

     > Typically X, Y are paths to the actual data

1. Define `read_data`:

   - Input: 

     - `x1, x2, ..., xn, y1, y2, ..., ym` - a tuple of input and output corresponding to the output of `list_data`. 

   - Output:

     - `x1, x2, ..., xn, y1, y2, ..., ym` - data to pass to model and loss function

   - Effects:

     Defines how to read actual data from listed data. Do any online preprocessing. This function will be called in `__getitem__`. The returned data type should be `torch.Tensor`. 

1. (Optional) Define `n_inputs` if it is not `1`. 

   > n_inputs and n_outputs (inferred) must be consistent between data and model

1. Define any parameter to model as a class attribute. 

1. Define model

   - `__init__(self, params from data, **kwargs)`
   - `forward(self, x1, x2, ..., xn)`

1. (Optional) Define loss function
