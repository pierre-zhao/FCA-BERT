BERT_BASE_DIR='./bert_base'
GLUE_DIR='./GLUE/dataset'
OUTPUT_DIR='./output_mask_add_one'

mkdir $OUTPUT_DIR
mkdir $OUTPUT_DIR/CoLA
mkdir $OUTPUT_DIR/SST-2 
mkdir $OUTPUT_DIR/RTE
mkdir $OUTPUT_DIR/QQP
mkdir $OUTPUT_DIR/QNLI
mkdir $OUTPUT_DIR/MNLI-m



python3 main.py --TASK "cola" \
--BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
--CHECKPOINT_PATH $BERT_BASE_DIR/bert_model.ckpt \
--VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
--DATA_DIR $GLUE_DIR/CoLA \
--EPOCH 5 \
--BATCH_SIZE 32 \
--LR_SOFT_EXTRACT 1.5e-3 \
--LR_BERT 2e-5 \
--LAMBDA 7e-3 \
--OUTPUT_DIR $OUTPUT_DIR/CoLA 



# python3 main.py --TASK "rte" \
# --BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
# --CHECKPOINT_PATH $BERT_BASE_DIR/bert_model.ckpt \
# --VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
# --DATA_DIR $GLUE_DIR/RTE  \
# --EPOCH 5 \
# --BATCH_SIZE 16 \
# --LR_SOFT_EXTRACT 3e-3 \
# --LR_BERT 3e-5 \
# --LAMBDA 2e-3 \
# --OUTPUT_DIR $OUTPUT_DIR/RTE 



# python3 main.py --TASK "sst-2" \
# --BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
# --CHECKPOINT_PATH $BERT_BASE_DIR/bert_model.ckpt \
# --VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
# --DATA_DIR $GLUE_DIR/SST-2 \
# --EPOCH 5 \
# --BATCH_SIZE 64 \
# --LR_SOFT_EXTRACT 5e-4 \
# --LR_BERT 3e-5 \
# --LAMBDA 2e-4 \
# --OUTPUT_DIR $OUTPUT_DIR/SST-2 



# python3 main.py --TASK "qqp" \
# --BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
# --CHECKPOINT_PATH $BERT_BASE_DIR/bert_model.ckpt \
# --VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
# --DATA_DIR $GLUE_DIR/QQP \
# --EPOCH 5 \
# --BATCH_SIZE 64 \
# --LR_SOFT_EXTRACT 1e-4 \
# --LR_BERT 3e-5 \
# --LAMBDA 3e-4 \
# --OUTPUT_DIR $OUTPUT_DIR/QQP 



# python3 main.py --TASK "qnli" \
# --BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
# --CHECKPOINT_PATH $BERT_BASE_DIR/bert_model.ckpt \
# --VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
# --DATA_DIR $GLUE_DIR/QNLI \
# --EPOCH 5 \
# --BATCH_SIZE 16 \
# --LR_SOFT_EXTRACT 2e-4 \
# --LR_BERT 3e-5 \
# --LAMBDA 1.5e-4 \
# --OUTPUT_DIR $OUTPUT_DIR/QNLI


# python3 main.py --TASK "mnli-m" \
# --BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
# --CHECKPOINT_PATH $BERT_BASE_DIR/bert_model.ckpt \
# --VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
# --DATA_DIR $GLUE_DIR/MNLI-m \
# --EPOCH 5 \
# --BATCH_SIZE 64 \
# --LR_SOFT_EXTRACT 2e-4 \
# --LR_BERT 3e-5 \
# --LAMBDA 1e-4 \
# --OUTPUT_DIR $OUTPUT_DIR/MNLI-m


################### finetune prediction #####################

# python3 main.py --TASK "cola" \
# --BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
# --CHECKPOINT_PATH $OUTPUT_DIR/CoLA/finetune.hdf5 \
# --VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
# --DATA_DIR $GLUE_DIR/CoLA \
# --PREDICT_ONLY \
# --PRED_GLUE \
# --BATCH_SIZE 128 \
# --MODEL_FORMAT "HDF5" \
# --OUTPUT_DIR $OUTPUT_DIR/CoLA



################### FCA prediction #####################

# python3 main.py --TASK "cola" \
# --BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
# --CHECKPOINT_PATH $OUTPUT_DIR/CoLA/finetune.weight_sum.hdf5 \
# --VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
# --DATA_DIR $GLUE_DIR/CoLA \
# --PREDICT_ONLY \
# --PRED_GLUE \
# --BATCH_SIZE 128 \
# --RETENTION_CONFIG "34, 33, 32, 32, 31, 30, 30, 30, 30, 29, 28, 28" \
# --MODEL_FORMAT "HDF5" \
# --OUTPUT_DIR $OUTPUT_DIR/CoLA

