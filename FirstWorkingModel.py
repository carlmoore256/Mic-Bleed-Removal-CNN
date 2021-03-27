experiment = Experiment(api_key="OKhPlin1BVQJFzniHu1f3K1t3",
                        project_name="micplacementwavenet", workspace="cm5409a")

batch_size = 32
steps_per_epoch = 1000
num_epochs = 20

# gen_obj_train = TrainGenerator(x_train, y_train, win_size, batch_size)
# gen_obj_val = TrainGenerator(x_val, y_val, win_size, batch_size)


# train_gen = gen_obj_train.generator()
# val_gen = gen_obj_val.generator()

# init_x, init_y = next(train_gen)

init_x = x_train[0]
init_y = y_train[0]

input_shape = (x_train.shape[1], 1)
output_shape = (y_train.shape[1], 1)
main_dim = input_shape[1]

#INPUT =======================================================
input_layer = Input(shape=input_shape)

encoded = Conv1D(filters=4, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(input_layer)
encoded = LeakyReLU(alpha=0.05)(encoded) # ACTIVATION
encoded = MaxPooling1D(pool_size=2)(encoded) # DOWNSAMPLE

encoded = Conv1D(filters=8, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(encoded)
encoded = LeakyReLU(alpha=0.05)(encoded) # ACTIVATION
encoded = MaxPooling1D(pool_size=2)(encoded) # DOWNSAMPLE

encoded = Conv1D(filters=16, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(encoded)
encoded = LeakyReLU(alpha=0.05) (encoded)# ACTIVATION
encoded = MaxPooling1D(pool_size=2)(encoded) # DOWNSAMPLE

encoded = Conv1D(filters=32, kernel_size=7, padding='same',
                 input_shape=(1, init_x.shape[1]))(encoded)
encoded = LeakyReLU(alpha=0.05) (encoded)# ACTIVATION
encoded = MaxPooling1D(pool_size=2)(encoded) # DOWNSAMPLE

encoded = Conv1D(filters=64, kernel_size=7, padding='same',
                 input_shape=(1, init_x.shape[1]))(encoded)
encoded = LeakyReLU(alpha=0.05) (encoded)# ACTIVATION
encoded = MaxPooling1D(pool_size=2)(encoded) # DOWNSAMPLE

encoded = Conv1D(filters=128, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(encoded)
encoded = LeakyReLU(alpha=0.05) (encoded)# ACTIVATION
encoded = MaxPooling1D(pool_size=2)(encoded) # DOWNSAMPLE

encoded = Conv1D(filters=256, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(encoded)
encoded = LeakyReLU(alpha=0.05) (encoded)# ACTIVATION
encoded = MaxPooling1D(pool_size=2)(encoded) # DOWNSAMPLE

# =====================================================

decoded = Conv1D(filters=64, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(encoded)
decoded = LeakyReLU(alpha=0.05)(decoded) # ACTIVATION
decoded = UpSampling1D(size=2)(decoded) # UPSAMPLE


decoded = Conv1D(filters=128, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(decoded)
decoded = LeakyReLU(alpha=0.05)(decoded) # ACTIVATION
decoded = UpSampling1D(size=2)(decoded) # UPSAMPLE


decoded = Conv1D(filters=64, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(decoded)
decoded = LeakyReLU(alpha=0.05)(decoded) # ACTIVATION
decoded = UpSampling1D(size=2)(decoded) # UPSAMPLE


decoded = Conv1D(filters=32, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(decoded)
decoded = LeakyReLU(alpha=0.05)(decoded) # ACTIVATION
decoded = UpSampling1D(size=2)(decoded) # UPSAMPLE

decoded = Conv1D(filters=16, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(decoded)
decoded = LeakyReLU(alpha=0.05)(decoded) # ACTIVATION
decoded = UpSampling1D(size=2)(decoded) # UPSAMPLE

decoded = Conv1D(filters=8, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(decoded)
decoded = LeakyReLU(alpha=0.05)(decoded) # ACTIVATION
decoded = UpSampling1D(size=2)(decoded) # UPSAMPLE

decoded = Conv1D(filters=2, kernel_size=1, padding='same',
                         input_shape=(init_y.shape[1], 1))(decoded)
decoded = LeakyReLU(alpha=0.05)(decoded) # ACTIVATION
decoded = UpSampling1D(size=2)(decoded) # UPSAMPLE



decoded = Dense(output_shape[1], activation='tanh')(decoded)

model = Model(input_layer, decoded)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()


# cb = callback(gen_obj_val, model)

# result = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                            #  epochs=num_epochs, validation_data=val_gen, validation_steps=100, verbose=1,
                            #  callbacks=[cb])

result = model.fit(x_train,
                   y_train,
                   batch_size=batch_size,
                   shuffle=True,
                   epochs=num_epochs,
                   validation_data=(x_val, y_val))
