主要分為監督式以及非監督


監督式的部份：

NormalSeq2Seq_V1.py為主程式，跑出來的結果跟論文內容一樣

RandomForest.py為特徵排序
Model_compare.py是將具有attention機制的Seq2Seq與沒有attention做比較



非監督的部份：


1. appscript.sh主要控制兩個套件→pcapplusplus以及CICFlowMeter


2. 其中pcapplusplus為封包切割工具，可選擇要以連線導向或封包導向做切割


3. CICFlowMeter則為特徵提取工具，是將一流量做特徵提取


4. COprcessing.py是針對經過CICFlowMeter輸出之csv檔進行資料前處理，將不必要的特徵丟棄，並將流量數目少於60之csv檔丟棄，高於60則刪減為60（但這樣做後來覺得不對，因為CICFlowMeter是流量導向的特徵提取，大部分論文切60的是指封包數目，而不是流量數目，要改進）

5. Model_test_forNoLabel.py是把預訓練過得模型進行即時的判斷


