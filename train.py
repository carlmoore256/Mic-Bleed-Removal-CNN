import model
import callback



cb = callback(x_val, y_val, model, num_tests=1)

result = model.fit(x_train, 
                   y_train, 
                   batch_size=batch_size,
                   shuffle=True,
                   epochs=num_epochs,
                   validation_data=(x_val, y_val),
                   callbacks=[cb])

model.save('/content/drive/My Drive/Datasets/MultitrackStems/models/oh_to_snare_5_epoch.h5')