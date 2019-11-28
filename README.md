**MyGeoMAN**

### Model Input
The model has the following inputs:
* local_inputs: the input of local spatial attention, shape->[batch_size, n_steps_encoder, n_input_encoder]
* global_inputs: the input of global spatial attention, shape->[batch_size, n_steps_encoder, n_sensors]
* external_inputs: the input of external factors, shape->[batch_size, n_steps_decoder, n_external_input]
* local_attn_states: shape->[batch_size, n_input_encoder, n_steps_encoder]
* global_attn_states: shape->[batch_size, n_sensors, n_input_encoder, n_steps_encoder]
* labels: ground truths, shape->[batch_size, n_steps_decoder, n_output_decoder]


#### 参数说明
* n_steps_encoder: length of encoder, i.e., how many historical time steps we use for predictions
* n_steps_decoder: length of decoder, i.e., how many future time steps we predict

