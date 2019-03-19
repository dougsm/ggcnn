#! /bin/bash

wget http://s.dougsm.com/ggcnn_rss/ggcnn_rss.tar.gz -P networks/
tar -xvzf networks/ggcnn_rss.tar.gz -C networks/
rm networks/ggcnn_rss.tar.gz

wget http://s.dougsm.com/ggcnn_rss/val_input.zip -P networks/ggcnn_rss/
unzip networks/ggcnn_rss/val_input.zip
rm networks/ggcnn_rss/val_input.zip
