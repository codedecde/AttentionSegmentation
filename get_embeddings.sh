#!/bin/bash

# usage: bash get_embeddings.sh --type=glove --dim=50
# usage: bash get_embeddings.sh --type=elmo

# Default arguments
EMBED_DIM=50
EMBED_TYPE=glove
EMBED_BASE_DIR=Data/embeddings

# Argument parsing
for i in "$@"
do
case $i in
	-t=*|--type=*)
	EMBED_TYPE="${i#*=}"
	shift
	;;
	-d=*|--dim=*)
	EMBED_DIM="${i#*=}"
	shift
	;;
	*)
	;;
esac
done

# Get embeddings based on parameters passed
if [ ${EMBED_TYPE} = "glove" ]; then
	echo ${EMBED_DIM}
	if [ ${EMBED_DIM} = "" ]; then
		EMBED_DIM=50
	fi
	EMBED_FILE=glove.6B.${EMBED_DIM}d.txt.gz
    EMBED_SRC=https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/${EMBED_FILE}
	EMBED_DIR=${EMBED_BASE_DIR}/GloveEmbeddings
	if [ ! -d ${EMBED_DIR} ]; then
		mkdir ${EMBED_DIR}
		touch ${EMBED_DIR}/.gitkeep
	fi
	DEST_EMBED_PATH=${EMBED_DIR}/${EMBED_FILE}
	echo "${DEST_EMBED_PATH}"
	if [ ! -f ${DEST_EMBED_PATH} ]; then
		echo "Downloading ${EMBED_FILE}"
	 	wget -O ${DEST_EMBED_PATH} ${EMBED_SRC}
	fi
elif [ ${EMBED_TYPE}="elmo" ]; then
	OPT_SRC=https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
	WEIGHTS_SRC=https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
	EMBED_DIR=${EMBED_BASE_DIR}/ELMOEmbeddings
	if [ ! -d ${EMBED_DIR} ]; then
		mkdir ${EMBED_DIR}
		touch ${EMBED_DIR}/.gitkeep
	fi
	OPT_FILE=${EMBED_DIR}/"elmo_2x4096_512_2048cnn_2xhighway_options.json"
	WEIGHTS_FILE=${EMBED_DIR}/"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
	if [ ! -f ${OPT_FILE} ]; then
		echo "Downloading ${OPT_FILE}"
		wget -O ${OPT_FILE} ${OPT_SRC}
	fi
	if [ ! -f ${WEIGHTS_FILE} ]; then
		echo "Downloading ${WEIGHTS_FILE}"
		wget -O ${WEIGHTS_FILE} ${WEIGHTS_SRC}
	fi
else
	echo "TYPE ${EMBED_TYPE} unrecognized"
fi
