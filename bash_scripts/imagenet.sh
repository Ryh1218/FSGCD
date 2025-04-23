# Change the path to your own environment
# For example, PYTHON='/root/miniconda3/envs/fsgcd/bin/python'
PYTHON={YOUR_PYTHON_PATH}
# Change the output path accordingly
# For example, BASE_DIR=/root/FSGCD/outs/
BASE_DIR={YOUR_OUTPUT_PATH}
PYTHON_SCRIPT="train"

export CUDA_VISIBLE_DEVICES=0
PYTHON_PATH="${PYTHON_SCRIPT}"

SAVE_DIR=${BASE_DIR}${PYTHON_SCRIPT}/
mkdir -p ${SAVE_DIR}

EXP_NUM=$(ls ${SAVE_DIR} | grep -c 'log.*\.txt')
EXP_NUM=$((${EXP_NUM}))

echo "Python script name: ${PYTHON_SCRIPT}" > ${SAVE_DIR}log${EXP_NUM}.txt

${PYTHON} -m ${PYTHON_PATH} \
            --dataset_name 'imagenet_100' \
            --batch_size 32 \
            --epochs 200 \
            --num_workers 16 \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --lr 0.1 \
            --log_dir ${SAVE_DIR} \
            --known_class 10 \
            --prop_train_labels 0.1 \
            --pretrain_epoch 120 \
>> ${SAVE_DIR}log${EXP_NUM}.txt