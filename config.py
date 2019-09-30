import jsonschema
import yaml

props = [
         ('training_path', 'string'),
         ('generated_path', 'string'),
         ('learning_rate', 'number'),
         ('validation_proportion', 'number'),
         ('validation_epochs_frequency', 'integer'),
         ('training_epochs', 'integer'),
         ('sampling_rate', 'integer'),
         ('batch_size', 'integer'),
         ('batches_report', 'integer'),
         ('snippets_per_audio_file', 'integer'),
         ('output_size', 'integer'),
         ('down_kernel_size', 'integer'),
         ('up_kernel_size', 'integer'),
         ('depth', 'integer'),
         ('num_filters', 'integer'),
         ('device', 'string'), # cpu or cuda
        ]

def config_schema():
    prop_schm = {}
    for p, t in props:
        prop_schm[p] = {'type': t}
    required = [p for p, _ in props]
    return {'type': 'object', 'properties': prop_schm, 'required': required}

def load(configpath):
    with open(configpath, 'r') as f:
        config_content = yaml.safe_load(f.read())
        jsonschema.validate(instance=config_content, schema=config_schema())
        return config_content
