#!/bin/sh

tensorboard --logdir tensorflow/trainingsummaries --port=8008 &

ARCHITECTURE="inception_v3"
python -m scripts.retraining \
   --bottleneck_dir=tensorflow/bottlenecks \
   --epochs=900 \
   --model_dir=tensorflow/models/ \
   --summaries_dir=tensorflow/trainingsummaries/"${ARCHITECTURE}" \
   --graph=tensorflow/retrained_graph.pb \
   --labels=tensortlow/retrained_labels.txt \
   --architecture="${ARCHITECTURE}" \
   --image_dir=tensorflow/argos \

