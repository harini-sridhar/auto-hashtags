DATASET=${1:-"CocoCaptions"}
DATASET_DIR=${2:-"/p/lscratchh/jjayaram/image-captioning/data"}
OUTPUT_DIR=${3:-"/p/lscratchh/jjayaram/image-captioning/results"}

WARMUP="4000"
LR0="512**(-0.5)"

python main.py \
  --save recurrent-attention \
  --dataset ${DATASET} \
  --dataset-dir ${DATASET_DIR} \
  --results-dir ${OUTPUT_DIR} \
  --model Img2Seq \
  --model-config "{'encoder': {'model': 'resnet50'}, \
                   'decoder': {'type': 'recurrent','num_layers': 3, 'dropout': 0.2}}" \
  --b 128 \
  --max-length 20 \
  --label-smoothing 0.1 \
  --trainer Img2SeqTrainer \
  --optimization-config "[{'step_lambda':
                          \"lambda t: { \
                              'optimizer': 'Adam', \
                              'lr': ${LR0} * min(t ** -0.5, t * ${WARMUP} ** -1.5), \
                              'betas': (0.9, 0.98), 'eps':1e-9}\"
                          }]"


# python main.py \
#   --save recurrent-attention \
#   --dataset ${DATASET} \
#   --dataset-dir ${DATASET_DIR} \
#   --results-dir ${OUTPUT_DIR} \
#   --model Img2Seq \
#   --model-config "{'encoder': {'model': 'resnet50'}, \
#                    'decoder': {'type': 'recurrent_attention','num_layers': 3, 'concat_attention': True,\
#                                'attention': {'mode': 'dot_prod', 'dropout': 0.1, 'output_transform': True, 'output_nonlinearity': 'relu'}}}" \
#   --b 128 \
#   --max-length 20 \
#   --label-smoothing 0.1 \
#   --trainer Img2SeqTrainer \
#   --optimization-config "[{'step_lambda':
#                           \"lambda t: { \
#                               'optimizer': 'Adam', \
#                               'lr': ${LR0} * min(t ** -0.5, t * ${WARMUP} ** -1.5), \
#                               'betas': (0.9, 0.98), 'eps':1e-9}\"
#                           }]"
