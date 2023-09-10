# TransTraffic: Predicting Network Traffic using Low Resource Data

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

<img width="940" alt="image" src="https://github.com/DSAIL-SKKU/TransTraffic-ICTC-2022/assets/60170358/14594bbc-c893-41c8-99ea-caf5c34328f8">

## Performance

<img width="481" alt="image" src="https://github.com/DSAIL-SKKU/TransTraffic-ICTC-2022/assets/60170358/131ffc66-53f7-4474-8c86-842295b3a342">

Our model outperforms both MLP and LSTM models in three different user activity data (i.e., *Bulk*, *Video*, and *Web*). In particular, compared to other baselines, the proposed model shows relatively good performance even with the extremely small amount of data (i.e., *Video* and *Web*). 

<img width="471" alt="image" src="https://github.com/DSAIL-SKKU/TransTraffic-ICTC-2022/assets/60170358/9588770b-6e68-44c9-9e6b-4c12a7a2e74c">

To explore the effectiveness of the transfer learning, we compare the traffic prediction performance of the proposed model between with and without transfer learning. To explore the effectiveness of the transfer learning, we compare the traffic prediction performance of the proposed model between with and without transfer learning.
