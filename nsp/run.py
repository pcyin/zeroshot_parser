import argparse
import datetime
import json
import logging
import os
import random
import torch
import _jsonnet
from slugify import slugify


from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_module_and_submodules
from torch import cuda


def _get_logger():
    DIR = os.path.dirname(__file__)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        os.path.join(DIR, '../logs/log_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('test_den_acc', 'dev_seq_acc', 'test_seq_acc',
                                                                'train_loss', 'best_val_loss', 'best_epoch',
                                                                'batch_size', 'lr', 'do'))
    return logger


def _run_experiment(config_file, serialization_dir, config_override, embeddings, cuda_ind, domain, learning_rate,
                    dropout):
    config_override["trainer"] = {"optimizer": {"lr": learning_rate}, "cuda_device": cuda_ind}
    num_train_examples = len(open(config_override["train_data_path"]).readlines())
    config = json.loads(_jsonnet.evaluate_file(config_file))
    batch_size = config['data_loader']['batch_sampler']['batch_size']
    max_epoch = config['trainer']['num_epochs']
    total_train_steps = num_train_examples // batch_size * max_epoch
    num_steps_per_epoch = num_train_examples // batch_size

    config_override['trainer'].setdefault('learning_rate_scheduler', {})['warmup_steps'] = total_train_steps * 0.1
    config_override['trainer'].setdefault('learning_rate_scheduler', {})['total_steps'] = total_train_steps
    # config_override['trainer'].setdefault('learning_rate_scheduler', {})['num_steps_per_epoch'] = total_train_steps

    or_model = {}
    if embeddings == 'elmo':
        or_model["source_embedder"] = {"elmo": {"dropout": dropout}}
    if domain is not None:
        or_model["domain"] = domain
    config_override["model"] = or_model

    train_model_from_file(parameter_filename=config_file,
                          serialization_dir=serialization_dir,
                          overrides=json.dumps(config_override),
                          force=True)


def _run_all(config_file, serialization_dir, scores_dir, config_override, embeddings, cuda_ind, domain, learning_rates,
             dropouts):
    """Runs an experiment for each hyperparams configuration and logs all results."""
    for learning_rate in learning_rates:
        for dropout in dropouts:
            # with torch.autograd.set_detect_anomaly(True):
            _run_experiment(config_file, serialization_dir, config_override, embeddings, cuda_ind, domain,
                            learning_rate, dropout)

            score = json.load(open(scores_dir, 'r'))
            test_den_acc = score.get('test_den_acc')
            dev_seq_acc = score.get('best_validation_seq_acc')
            test_seq_acc = score.get('test_seq_acc')
            best_epoch = score.get('best_epoch')
            train_loss = score.get('training_loss')
            best_validation_loss = score.get('best_validation_loss')
            logger.info('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(test_den_acc, dev_seq_acc,
                                                                    test_seq_acc, train_loss, best_validation_loss,
                                                                    best_epoch, learning_rate, dropout))


def _parse_args():
    parser = argparse.ArgumentParser(
      description='experiment parser.',
      formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--embeddings', '-e', default='glove', choices=['glove', 'elmo', 'bert'],
                        help='Pretrained embeddings to use (options: [glove, elmo]).')
    parser.add_argument('--version', '-v', default='nat',
                        # choices=['nat', 'lang', 'granno', 'overnight', 'nat_overnight'],
                        help='Training version to use (options: [nat, lang, granno, overnight]).')
    parser.add_argument('--domain', '-d', default='geo', choices=['geo', 'scholar'],
                        help='(options: [geo, scholar]).')
    parser.add_argument('--label', default='', type=str)
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()
    embeddings = args.embeddings
    domain = args.domain
    version = args.version

    scores_dir = "tmp/output/metrics.json"
    if embeddings == 'elmo':
        config_file = "configs/config_elmo.json"
    elif embeddings == 'glove':
        config_file = "configs/config_glove.json"
    else:
        config_file = "configs/config_bert.json"

    config_file = "configs/config_bert.json"

    num_devices_available = torch.cuda.is_available()
    print('num_devices_available={}'.format(num_devices_available))
    config_override = dict()
    cuda_ind = 0 if num_devices_available > 0 else -1  # train on gpu, if possible

    train_file = config_override["train_data_path"] = "data/train_{}_{}.json".format(domain, version)
    config_override["validation_data_path"] = "data/dev_{}_{}.json".format(domain, version)
    config_override["test_data_path"] = "data/test_{}.json".format(domain)
    # config_override.setdefault('trainer', {})['num_epochs'] = 2

    serialization_dir = f"tmp/{slugify(train_file)}{args.label}"

    random.seed(0)
    import_module_and_submodules('nsp')

    logger = _get_logger()

    # hyper-params to search over
    if embeddings == 'elmo':
        dropouts = [0.4, 0.0, 0.1, 0.2, 0.3, 0.5]
    else:
        dropouts = [0.0]
    #learning_rates = [0.01, 0.007, 0.013]
    #learning_rates = [1e-3]
    learning_rates = [3e-5]

    _run_all(config_file, serialization_dir, scores_dir, config_override, embeddings, cuda_ind, domain, learning_rates,
             dropouts)
