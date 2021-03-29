import numpy as np
import matplotlib.pyplot as plt

class callback(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, model, num_tests=1, audio_preview=True, sr=44100):
        self.losses = []
        self.model = model
        self.x_val = x_val
        self.y_val = y_val
        self.num_examples = x_val.shape[0]
        self.num_tests = num_tests
        self.sr = sr
        self.audio_preview = audio_preview

        self.difference_mask = False
        
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return

    def random_sample(self):
        rand_idx = np.random.randint(0, high=self.num_examples)
        return self.x_val[rand_idx, :, :], self.y_val[rand_idx, :, :]
 
    def on_epoch_end(self, epoch, logs={}):
        for _ in range(self.num_tests):
          x, y = self.random_sample()
          y_p = self.model.predict(x.reshape((1,x.shape[0],1)))
          x = np.squeeze(x)
          y = np.squeeze(y)
          y_p = np.squeeze(y_p)

          if self.difference_mask:
            y = x + y
            y_p = x + y_p

          print('x/y_p diff:')
          print(abs(np.sum(x) - np.sum(y_p)))

          print('x vs predicted y')
          plt.plot(x, color='red')
          plt.plot(y_p)
          plt.show()

          print('ground truth y vs predicted y')
          plt.plot(y, color='red')
          plt.plot(y_p)
          plt.show()

          if self.audio_preview:
            print('input sample:')
            ipd.display(ipd.Audio(x, rate=self.sr, autoplay=False))
            print('ground truth:')
            ipd.display(ipd.Audio(y, rate=self.sr, autoplay=False))
            print('prediction')
            ipd.display(ipd.Audio(y_p, rate=self.sr, autoplay=False))

        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return