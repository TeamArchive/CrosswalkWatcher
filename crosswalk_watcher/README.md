# Crosswalk Watcher
<pre>
( Open API : UTIC )  
  ├→( Patton Recognition Model )  
  │   └→( Web Server )  
  └───────┴→( Client : 중앙통제센터 ) → 사람이 확인 
                  └→ 해당 관할 112, 119 전파 후 조치
</pre>

## Patton Recognition Model
  1. Object detection : Yolo v5
  2. Detected object tracking : DeepSort
  3. Abnormal detection from CCTV video

<pre>
( Cropped Image ) -→ Inception v3
				↑				  ↓ 	   ⤺
( YOLO v5 + DeepSORT )	  	  FC Layer -→ LSTM -→ FC Layer -→ Activation -→ ( Output )
				↓				  ↑		 
( Label, Anchor Box ) -→ MLP 
</pre>  

### data support
* UTIC; 경찰청 교통정보(교통정보 재공처) : <http://www.utic.go.kr>

* ref (1) : <https://www.koreascience.or.kr/article/JAKO201914260900658.pdf>
* ref (2) : <https://jeinalog.tistory.com/26>
* ref (3) : <https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/>
* ref (4) : <http://www.kibme.org/resources/journal/20190814141103239.pdf>
