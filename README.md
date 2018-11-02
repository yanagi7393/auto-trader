Auto_trading (Here are 2 models)
========================
AE_CNN_ArLSTM model
------------------------
#### 0_AE_CNN_ArLSTM_model is AutoEncoder-Convolutional NN-AutoRegression LSTM model. This model is based on predicting next price (1min later, 5min later or your setting).
 
	1. From the collected ticker data, calculate the technical indicator and use it as an input.
	2. Input Calculated 64 indicators for each point (× 60 sequence batch) to CNN.
	3. Create a compressed feature vector in the learning of the AutoEncoder structure which have Conv-DeConv.
	4. Extracted 60 sequence featrures will be devided into feature[0:58] and feature[59] (this will be used as a label)
	5. Training is performed to predict the 59th feature with the input feature[0:58]. input Feature[0:58] -> predict Feature[59]
	6. After the training, when predicting process, future feature of next time will be predicted with input feature[1:59]. input Feature[1:59] -> predict Feature[60]  
	  
	loss =  
	  1. AutoEncoder reconstruction loss. (To make features which are compressed but meaningful)
	  2. CNN feature[0:58]->LSTM->output feature <-> CNN feature[59] (MSE, Cross-entropy) loss. #approximate vector's value and distribution.
	  3. CNN feature[0:59]->PriceNet->output price <-> label[0:59] (MSE, Cross-entropy) loss.
	  4. CNN feature[0:58]->LSTM->ourput feature->PriceNet->output price <-> label[59] (MSE, Cross-entropy) loss. 

* **After the training, PREDICT PRICE = input[0:59]->CNN feature[1:59]->LSTM->output feature(predicted)->PriceNet->output price(predicted)**  
![model_1](/model_1.png)
  
CNN_(Ar)BiLSTM_DDDQN model
--------------------------
#### 1_CNN_BiLSTM_DDDQN_model is Convolutional NN-Bidirectional(AutoRegression or not) LSTM-DoublingDuelingDQN model. This model is based on action-value ([buy, look, sell] or with percent e.g.[100%buy,50%buy,25%buy,look,25%sell,50%sell,100%sell]).

	1. From the collected ticker data, calculate the technical indicator and use it as an input this will be used as states.
	2. Agent contacts the environment, returning states (indicator values of 60 periods from past to present), action, and reward. Also returns the history of [Now have money/Total money Ratio, Action, Action Percentage] for meta learning.
	3. Input Calculated 64 indicators for each point (× item counts, × 60 sequence batch) to CNN.
	4. input concatenate[CNN feature[0:59] + history vector(for meta learning)] into BiLSTM1 and BiLSTM2.
	5. BiLSTM1 is the route for the action-value. BiLSTM2 is the route for the price prediction & the action-value.
	   (BiDRw QN model's BiLSTM2 predict price directly)  
	   (BiDRx QN model's BiLSTM2 is similar with AE_CNN_ArLSTM_model's LSTM predict method // CNN feature[0:58]->BiLSTM2->approximate to CNN feature[59], and then CNN feature[1:59]->BiLSTM2->predict next feature)  
	6. concatenate[BiLSTM1 output[batch,-1,:], BiLSTM2 output[batch,-1,:]] (which output[:,-1,:] are unified sequences), and then which is devided into action & value (dueling).
	7. Agent exploit & explore in the environment, and train action-value with doubling technique by Target Q-value[Main action].  
	  
	loss =
	  1. Doubling Dueling DQN loss (MSE(targetQ[s1], mainQ[s])).
	  2. BiLSTM2->output feature->FC->predict price <-> price label (MSE) loss.
	  3. (BiDRx QN model only) CNN feature[0:58]->BiLSTM2->output feature <-> CNN feature[59] (MSE, Cross-entropy) loss. #approximate vector's value and distribution.

* **After the training, ACTION = input[0:59]->CNN feature[0:59]->BiLSTM1, BiLSTM2->concat[output1, output2]->Action-Value Q-> Action which is argmax(Q)**  
![model_2](/model_2.png)

Training & Trading step
========================
1. collect ticker data (Here I attach, 'bainance' ticker data collecting code).
2. Training main model (You can choose the model 0_AE_CNN_ArLSTM_model or 1_CNN_BiLSTM_DDDQN_model)
3. Auto Trading & Back Testing (You have to perform a backtest to determine the threshold(for trading))
#
* **Collecting ticker data is performed in the following order.**  
Basically, 60sec(1min) candle is made and saved every 10 seconds (0 to 60, 10 to 70, 20 to 80, ..., 50 to 110 sec) from ticker data. In other words, if you collect data for one minute, you have six 1minute candles with 10 second intervals.

  1. First, if you run the code, last 6000 ticker data will be automatically downloaded from Cryptowatch. 2. Second, afterwards, it will automatically shift to download real-time ticker data from Bainance.
  
  2. If you re-execute the code, it will download the ticker data from time of code stopped until the current time by the Cryptowatch, and then, it will automatically shift to download real-time ticker data from Bainance.
#
* **Training main model**
  1. First, you should to make dataset. For making dataset, you need to make concatenated csv file using collected ticker data.
  2. For making dataset, you should to do pre-processing.
  3. Finally, you can make to train the model.
#
* **Back Testing & Real-time Testing**
  1. Before auto trading, you should to test whether trained model works or not. For backtest, First, you should to do backtest pre-processing. For real-time testing, you dont need to do pre-processing. The preprocessing will be performed in real time (every 10 sec intervals).
  2. Control the parameters & switches. And then, do back & real-time testing.
#
* **Auto Trading**
  1. For auto trading, you should to determine the trading conditions such as threshold which you can get through back testing.
  2. If the conditions are set, now you can do auto trading! (Auto trading will be performed every 10 seconds intervals)
