# TransTraffic: Predicting Network Traffic using Low Resource Data**

In private 5G/6G networks, an adequate and accurate resource management is essential. We propose a traffic prediction model, TransTraffic, that utilizes transfer learning for low resource data. Our evaluation demonstrates that leveraging prior knowledge from a similar traffic domain helps predict network traffic for a new domain or service.

This codebase contains the python scripts for the model for the ICTC 2022.  https://ieeexplore.ieee.org/abstract/document/9952575/

## Data

To learn the proposed model, we utilize an open source *[traffic prediction dataset](https://www.sciencedirect.com/science/article/pii/S1389128620312081).*

The dataset consists of network packets, collected from February 20th to October 6th in 2019, with the following components: the timestamp of the packet, protocol, the size of the payload, source/destination IP address, source/destination UDP/TCP port, and the types of the user activity (i.e., Interactive, Bulk, Video, and Web). The four types of user activity are summarized as follows.

- *Interactive*: Interactive activity includes traffic from real time interactive applications such as chatting app or remote file editing in Google Docs.
- *Bulk*: Bulk data transfer activity consists of traffic to applications that use significant portions of the network’s bandwidth for large data file transfers, e.g., large file downloads from Dropbox.
- *Video*: Video playback activity includes traffic from applications consuming videos (e.g., from Twitch or YouTube).
- *Web*: Web browsing consists of traffic for all activities within a web page such as downloading images or ads.

## Run
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/057e347f-38ce-4eff-9082-9d90bd6419e6/4d597967-2216-48b9-88d6-12428861da2e/Untitled.png)

## Performance
## 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/057e347f-38ce-4eff-9082-9d90bd6419e6/4baa34a3-fe7f-4a4b-9faf-75c8f187efc0/Untitled.png)

Our model outperforms both MLP and LSTM models in three different user activity data (i.e., *Bulk*, *Video*, and *Web*). In particular, compared to other baselines, the proposed model shows relatively good performance even with the extremely small amount of data (i.e., *Video* and *Web*). 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/057e347f-38ce-4eff-9082-9d90bd6419e6/f70f6ee0-5452-4735-adf0-79abd3857a39/Untitled.png)

To explore the effectiveness of the transfer learning, we compare the traffic prediction performance of the proposed model between with and without transfer learning. To explore the effectiveness of the transfer learning, we compare the traffic prediction performance of the proposed model between with and without transfer learning.
