# Train model


models = model.AbnormalDetectionModels( device )

def train(
	batch_size,
	input_tensor, target_tensor,
	enc_model, dec_model, video_model,
	enc_optim, dec_optim, video_optim,
	criterion, max
):
	enc_h 	= enc_model.init_hidden(batch_size)
	video_h = video_model.init_hidden(batch_size)
	dec_h 	= dec_model.init_hidden(batch_size)

	enc_optim.zero_grad()
	video_optim.zero_grad()
	dec_optim.zero_grad()

	# enc_model.zero_grad()
	# video_model.zero_grad()
	# dec_model.zero_grad()

	# < Encoding >

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	loss = 0

	# !TODO : video input loop
	# for video_ei in range(video_input_length):

	for ei in range(input_length):
		enc_outputs, enc_hidden = enc_model(
			input_tensor[ei], enc_h
		)

		enc_outputs[ei] = enc_outputs[0, 0]	# !TODO Chnage here

	# < Decoding >



def train_iters(
	device, 
	epochs, 
	batch_size, 
	train_loader, 
	val_loader,
	models,
	max_grad_norm = MAX_GRAD_NORM,
	learning_rate = LEARNING_RATE,
	coder_optim_func = optim.Adam,
	video_optim_func = optim.Adam,
	critical_func = nn.CrossEntropyLoss
):
	counter = 0
	criterion = critical_func()
	
	enc_optim 	= coder_optim_func(models.encoder.parameters(), lr=learning_rate)
	video_optim	= video_optim_func(models.video_lstm.parameters(), lr=learning_rate)
	dec_opttim	= coder_optim_func(models.decoder.parameters(), lr=learning_rate)

	model.to(device)

	model.train()
	for i in range(0, epochs):
		
    
		for inputs, labels in train_loader:
			counter += 1
			h = tuple([e.data for e in h])
			inputs, labels = inputs.to(device), labels.to(device)

			model.zero_grad()

			output, h = model(inputs, h)
			loss = criterion(output.squeeze(), labels.float())
			loss.backward()
			
			nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optimizer.step()





def train(
	device, 
	epochs, 
	batch_size, 
	train_loader, 
	val_loader,
	model = model.AbnormalDetectionModels,
	max_grad_norm = MAX_GRAD_NORM,
	learning_rate = LEARNING_RATE,
	optim_func = optim.Adam,
	critical_func = nn.CrossEntropyLoss
):
	counter = 0
	criterion = critical_func()
	optimizer = optim_func(model.parameters(), lr=learning_rate)

	model.to(device)

	model.train()
	for i in range(0, epochs):
		h = model.init_hidden(batch_size)
    
		for inputs, labels in train_loader:
			counter += 1

			h = tuple([e.data for e in h])
			inputs, labels = inputs.to(device), labels.to(device)

			model.zero_grad()

			output, h = model(inputs, h)
			loss = criterion(output.squeeze(), labels.float())
			loss.backward()
			
			nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optimizer.step()

			if counter % PRINT_PER == 0:
				val_h = model.init_hidden(batch_size)
				val_losses = []
				model.eval()

				for inp, lab in val_loader:
					val_h = tuple([each.data for each in val_h])
					inp, lab = inp.to(device), lab.to(device)
					out, val_h = model(inp, val_h)
					val_loss = criterion(out.squeeze(), lab.float())
					val_losses.append(val_loss.item())
					
				model.train()
				print("Epoch: {}/{}...".format(i+1, epochs),
						"Step: {}...".format(counter),
						"Loss: {:.6f}...".format(loss.item()),
						"Val Loss: {:.6f}".format(np.mean(val_losses)))

				if np.mean(val_losses) <= valid_loss_min:
					torch.save(model.state_dict(), './state_dict.pt')
					print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
					valid_loss_min = np.mean(val_losses)