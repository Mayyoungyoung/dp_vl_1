# Factored route reliability: complete comparison

| Split / arm | Top1 % | Brier | NLL | ECE % | K4 all-valid % | K4 valid modes |
|---|---:|---:|---:|---:|---:|---:|
|paired_dev/single|93.71|0.1390|0.4369|12.61|81.29|3.466|
|paired_dev/joint|93.75|0.1412|0.4420|11.96|80.48|3.468|
|paired_dev/marginal|91.74|0.2252|0.7310|25.19|76.93|3.346|
|paired_dev/conditional|93.13|0.1960|0.6458|22.37|79.90|3.392|
|paired_dev/conditional_endpoint|93.87|0.1971|0.6534|22.85|80.29|3.399|
|old_dev/single|94.75|0.1163|0.3765|9.15|79.63|3.327|
|old_dev/joint|95.37|0.1181|0.3796|8.72|79.32|3.318|
|old_dev/marginal|93.83|0.1166|0.3775|10.89|80.25|3.324|
|old_dev/conditional|94.14|0.1157|0.3747|11.30|78.70|3.321|
|old_dev/conditional_endpoint|94.75|0.1113|0.3632|11.42|78.40|3.327|
|dev32/single|93.75|0.1328|0.4242|7.62|79.51|3.316|
|dev32/joint|92.59|0.1358|0.4333|7.99|78.82|3.318|
|dev32/marginal|91.67|0.1355|0.4310|9.63|75.81|3.288|
|dev32/conditional|92.25|0.1328|0.4229|9.78|75.58|3.292|
|dev32/conditional_endpoint|93.52|0.1310|0.4179|10.31|79.17|3.315|
