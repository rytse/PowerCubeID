import tensorflow as tf
import os

from object_detection.utils import dataset_util

LABEL_DICT = {
	'powercube': 1
}

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example(fn, boxes):
	height = 1920	# TODO check this
	width = 1080
	image_format = 'png'
	
	with tf.gfile.GFile(fn, 'rb') as fid:
		encoded_image = fid.read()

	xmins = []
	xmaxs = []
	ymins = []
	ymaxs = []

	classes_text = []
	classes = []

	# Box should be in the form [x, y, w, h, class]
	for box in boxes:
		xmins.append(float(box[0] / width))
	        xmaxs.append(float((box[0] + box[2]) / width))
	        ymins.append(float(box[1] / height))
	        ymaxs.append(float((box[1] + box[3]) / height))
	        classes_text.append(box[4])
		classes.append(int(LABEL_DICT[box[4]]))

	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(fn),
		'image/source_id': dataset_util.bytes_feature(fn),
		'image/encoded': dataset_util.bytes_feature(encoded_image),
		'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))

	return tf_example

def main(_):
	writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

	i = 1 
	with open('../res/training_data/labels.csv', 'r') as fi:
		for line in fi:
			data = line.split(',')
			data = [int(d) for d in data]
			data.append('powercube')
			
			
			fn = '../res/training_data/all/output_' + '{:04d}'.format(i) + '.png'
			tf_example = create_tf_example(fn, [data])
			
			writer.write(tf_example.SerializeToString())
			
			i += 1

	writer.close()

if __name__ == '__main__':
	tf.app.run()
