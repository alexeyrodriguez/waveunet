
import yaml

# TODO validate
# https://stackoverflow.com/questions/3262569/validating-a-yaml-document-in-python
#  schema = Map({'training_path': Str(),
#                'generated_path': Str(),
#                'learning_rate': Float(),
#                'validation_proportion': Float(),
#                'sampling_rate': Int(),
#                'batch_size': Int(),
#                'batches_report': Int(),
#               })

def load(configpath):
    with open(configpath, 'r') as f:
        try:
            return yaml.safe_load(f.read())
        except YAMLError as error:
                print(error)
