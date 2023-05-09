MAMS datasets consist of 3 txt file with data structure like this:

```
The food was good, but it's not worth the wait--or the lousy service.foodpositive
The food was good, but it's not worth the wait--or the lousy service.servicenegative
```
 Add None samples for this sentence using data_process_ABSA.py (basically for txt files):
 ```
The food was good, but it's not worth the wait--or the lousy service.staffnone
The food was good, but it's not worth the wait--or the lousy service.menunone
The food was good, but it's not worth the wait--or the lousy service.ambiencenone
The food was good, but it's not worth the wait--or the lousy service.placenone
 ```

ACOS datasets consist of 3 tsv files with structure like this:
```
seems unlikely but whatever , i ' ll go with it .	-1,-1 LAPTOP#GENERAL 1 -1,-1
```
 Add None samples for this sentence using data_process_ACOS.py (basically for tsv files):
```
seems unlikely but whatever , i ' ll go with it .pricenone
seems unlikely but whatever , i ' ll go with it .design featuresnone
seems unlikely but whatever , i ' ll go with it .connectivitynone
seems unlikely but whatever , i ' ll go with it .usabilitynone
seems unlikely but whatever , i ' ll go with it .operation performancenone
seems unlikely but whatever , i ' ll go with it .portabilitynone
seems unlikely but whatever , i ' ll go with it .qualitynone
seems unlikely but whatever , i ' ll go with it .miscellaneousnone
```
