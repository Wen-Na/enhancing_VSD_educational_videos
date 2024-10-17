# Enhancing Saliency Detection in Educational Videos

This repository contains the source code that we used in our work "Enhancing Saliency Detection in Educational Videos". Specifically, we include the source code for evaluation metrics to test the performance of saliency detection models in our educational dataset. These metrics are based on the Python implementation offered by the maintainers of the [MIT/Tuebingen Benchmark](https://saliency.tuebingen.ai/).

In our study we applied the ViNet [1] saliency detection approach to our educational videos.
We used four metrics: Pearson’s correlation coefficient (CC), distribution similarity (SIM), normalized scanpath saliency (NSS), and Judd area under the curve  variant (AUC-J).


## Video saliency detection approach

The source code that implements each video saliency detection approach is provided by the authors and it is openly available:
- ViNet [1]: [github.com/samyak0210/ViNet](https://github.com/samyak0210/ViNeta)

## Dataset
- Educational dataset [2] and [3]: [https://osf.io/m7gj4/](https://osf.io/m7gj4/) and [https://osf.io/ptj75/](https://osf.io/ptj75/)


## References
[1] Samyak Jain, Pradeep Yarlagadda, Shreyank Jyoti, Shyamgopal Karthik, Ramanathan Subramanian, and Vineet Gandhi. Vinet: Pushing the limits of visual modality for audio-visual saliency prediction. In IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2021, Prague, Czech Republic, September 27 - Oct. 1, 2021, pages 3520–3527. IEEE, 2021.

[2] Jens Madsen, Sara U. J ́ulio, Pawel J. Gucik, Richard Steinberg, and Lucas C. Parra. Synchronized eye movements predict test scores in online video education. Proceedings of the National Academy of Sciences, 118(5):e2016980118, 2021.

[3] H. Zhang, K. M. Miller, X. Sun, and K. S. Cortina. Wandering eyes: eye movements during mind wandering in video lectures. Applied Cognitive Psychology, 34:449–464, 2020.



