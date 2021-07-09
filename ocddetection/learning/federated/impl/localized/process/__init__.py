from typing import Callable

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated.impl.localized import client


MODEL_FN = Callable[[], tff.learning.Model]
OPTIMIZER_FN = Callable[[], tf.keras.optimizers.Optimizer]
CLIENT_STATE_FN = Callable[[], client.State]
CLIENT_UPDATE_FN = Callable[[tf.data.Dataset, client.State, Callable, Callable, Callable], client.Output]
VALIDATION_FN = Callable[[tf.data.Dataset, client.State, tff.learning.ModelWeights, tff.learning.Model], client.Validation]
EVALUATION_FN = Callable[[tf.data.Dataset, client.State, tff.learning.ModelWeights, tff.learning.Model], client.Evaluation]


def __update_client(
  dataset: tf.data.Dataset,
  state: client.State,
  model_fn: MODEL_FN,
  optimizer_fn: OPTIMIZER_FN,
  update_fn: CLIENT_UPDATE_FN
) -> client.Output:
  return update_fn(
    dataset,
    state,
    model_fn,
    optimizer_fn
  )


def iterator(
  model_fn: MODEL_FN,
  client_state_fn: CLIENT_STATE_FN,
  client_optimizer_fn: OPTIMIZER_FN
):
  model = model_fn()
  client_state = client_state_fn()

  init_tf = tff.tf_computation(
    lambda: ()
  )
  
  server_state_type = init_tf.type_signature.result
  client_state_type = tff.framework.type_from_tensors(client_state)
  dataset_type = tff.SequenceType(model.input_spec)
  
  update_client_tf = tff.tf_computation(
    lambda dataset, state: __update_client(
      dataset,
      state,
      model_fn,
      client_optimizer_fn,
      tf.function(client.update)
    ),
    (dataset_type, client_state_type)
  )
  
  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(dataset_type)
  federated_client_state_type = tff.type_at_clients(client_state_type)

  def init_tff():
    return tff.federated_value(init_tf(), tff.SERVER)
  
  def next_tff(server_state, datasets, client_states):
    outputs = tff.federated_map(update_client_tf, (datasets, client_states))
    metrics = model.federated_output_computation(outputs.metrics)

    return server_state, metrics, outputs.client_state

  return tff.templates.IterativeProcess(
    initialize_fn=tff.federated_computation(init_tff),
    next_fn=tff.federated_computation(
      next_tff,
      (federated_server_state_type, federated_dataset_type, federated_client_state_type)
    )
  )


def __validate_client(
  dataset: tf.data.Dataset,
  state: client.State,
  model_fn: MODEL_FN,
  validation_fn: VALIDATION_FN
) -> client.Validation:
  return validation_fn(
    dataset,
    state,
    model_fn
  )


def validator(
  model_fn: MODEL_FN,
  client_state_fn: CLIENT_STATE_FN
):
  model = model_fn()
  client_state = client_state_fn()

  dataset_type = tff.SequenceType(model.input_spec)
  client_state_type = tff.framework.type_from_tensors(client_state)

  validate_client_tf = tff.tf_computation(
    lambda dataset, state: __validate_client(
      dataset,
      state,
      model_fn,
      tf.function(client.validate)
    ),
    (dataset_type, client_state_type)
  )

  federated_dataset_type = tff.type_at_clients(dataset_type)
  federated_client_state_type = tff.type_at_clients(client_state_type)    

  def validate(datasets, client_states):
    outputs = tff.federated_map(validate_client_tf, (datasets, client_states))
    metrics = model.federated_output_computation(outputs.metrics)

    return metrics

  return tff.federated_computation(
    validate,
    (federated_dataset_type, federated_client_state_type)
  )


def __evaluate_client(
  dataset: tf.data.Dataset,
  state: client.State,
  model_fn: MODEL_FN,
  evaluation_fn: EVALUATION_FN
) -> client.Evaluation:
  return evaluation_fn(
    dataset,
    state,
    model_fn
  )


def evaluator(
  model_fn: MODEL_FN,
  client_state_fn: CLIENT_STATE_FN
):
  model = model_fn()
  client_state = client_state_fn()

  dataset_type = tff.SequenceType(model.input_spec)
  client_state_type = tff.framework.type_from_tensors(client_state)

  evaluate_client_tf = tff.tf_computation(
    lambda dataset, state: __evaluate_client(
      dataset,
      state,
      model_fn,
      tf.function(client.evaluate)
    ),
    (dataset_type, client_state_type)
  )

  federated_dataset_type = tff.type_at_clients(dataset_type)
  federated_client_state_type = tff.type_at_clients(client_state_type)    

  def evaluate(datasets, client_states):
    outputs = tff.federated_map(evaluate_client_tf, (datasets, client_states))
    
    confusion_matrix = tff.federated_sum(outputs.confusion_matrix)
    aggregated_metrics = model.federated_output_computation(outputs.metrics)
    collected_metrics = tff.federated_collect(outputs.metrics)

    return confusion_matrix, aggregated_metrics, collected_metrics

  return tff.federated_computation(
    evaluate,
    (federated_dataset_type, federated_client_state_type)
  )
