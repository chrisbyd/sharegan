from paths import LOGS
import argparse

parser=argparse.ArgumentParser(description="train or eval a model on CIFAR or mnist")
parser.add_argument("--model_name",type=str,required=True)
#optimization parameters
parser.add_argument("--actions",required=True,type=str)

#parser.add_argument("--num_classes",required=True,type=int)
parser.add_argument("--dataset",default='cifar10',choices=['mnist','cifar10','cifar100'])
parser.add_argument('--image_size',default=32,type=int)

parser.add_argument('--view_pic_step',default=1000,type=int)
parser.add_argument('--print_step',default=100,type=int)
parser.add_argument('--ckpt_step',default=1000,type=int)
parser.add_argument('--summary_step',default=1000,type=int)
parser.add_argument('--inception_step',default=5000,type=int)
parser.add_argument('--inception_file',default="",type=str)
parser.add_argument("--num_generated_batches", default=500, type=int)

parser.add_argument('--batch_size',default=128,type=int)
parser.add_argument('--noise_dim',default=96,type=int)
parser.add_argument('--optimizer',default='adam',choices=['adam','rmsprop','sgd'])
parser.add_argument('--bn_decay',default=0.9,type=float)
parser.add_argument('--learning_rate',default=1e-3,type=float)
parser.add_argument('--adam_beta1',default=0.0,type=float)
parser.add_argument('--adam_beta2',default=0.9,type=float)
parser.add_argument('--max_iterations',default=100000,type=int)
parser.add_argument('--lr_decay',default=False,action="store_true")

args=parser.parse_args()
from logging_config import get_logging_config