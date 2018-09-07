import tensorflow as tf
import datetime


def inference_test(saved_model_dir, eval_data, signature="predict", repeat=10):
  tf.logging.set_verbosity(tf.logging.ERROR)

  time_start = datetime.utcnow()

  predictor = tf.contrib.predictor.from_saved_model(
      export_dir=saved_model_dir,
      signature_def_key=signature
  )
  time_end = datetime.utcnow()

  time_elapsed = time_end - time_start

  print ""
  print("Model loading time: {} seconds".format(time_elapsed.total_seconds()))
  print ""

  time_start = datetime.utcnow()
  output = None
  for i in range(repeat):
    output = predictor(
        {
            'input_image': eval_data[:10]
        }
    )

  time_end = datetime.utcnow()

  print "Prediction produced for {} instances repeated {} times".format(len(output['class_ids']), repeat)
  print ""

  time_elapsed = time_end - time_start
  print("Inference elapsed time: {} seconds".format(time_elapsed.total_seconds()))
  print ""

  print "Prediction output for the last instance:"
  for key in output.keys():
    print "{}: {}".format(key,output[key][0])
