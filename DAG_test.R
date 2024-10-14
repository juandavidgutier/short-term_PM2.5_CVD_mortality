library(ggdag)
library(dagitty)
library(lavaan)
library(CondIndTests)
library(dplyr)
library(GGally)
library(tidyr)
library(MKdescr)


#implied Conditional Independencies
dataset <- read.csv("D:/data.csv")
dataset <- dplyr::select(dataset, BC, DMS, PM, PM25, OC, SO2, SO4, Temperature, diabetes_prev, hypertension_prev, lead1_PM25)

#sd units
dataset$BC <- zscore(dataset$BC, na.rm = TRUE)
dataset$DMS <- zscore(dataset$DMS, na.rm = TRUE)
dataset$PM <- zscore(dataset$PM, na.rm = TRUE)
dataset$PM25 <- zscore(dataset$PM25, na.rm = TRUE)
dataset$OC <- zscore(dataset$OC, na.rm = TRUE)
dataset$SO2 <- zscore(dataset$SO2, na.rm = TRUE)
dataset$SO4 <- zscore(dataset$SO4, na.rm = TRUE)
dataset$Temperature  <- zscore(dataset$Temperature , na.rm = TRUE)
dataset$diabetes_prev <- zscore(dataset$diabetes_prev, na.rm = TRUE)
dataset$hypertension_prev <- zscore(dataset$hypertension_prev, na.rm = TRUE)
dataset$lead1_PM25 <- zscore(dataset$lead1_PM25, na.rm = TRUE)

#complete cases
dataset <- dataset[complete.cases(dataset), ] 

#DAG 
dag <- dagitty('dag {
BC -> DMS
BC -> PM
BC -> OC
BC -> SO2
BC -> SO4
BC -> diabetes_prev
BC -> hypertension_prev

DMS -> PM
DMS -> OC
DMS -> SO2
DMS -> SO4
DMS -> diabetes_prev
DMS -> hypertension_prev

PM -> OC
PM -> SO2
PM -> SO4
PM -> diabetes_prev
PM -> hypertension_prev

OC -> SO2
OC -> SO4
OC -> diabetes_prev
OC -> hypertension_prev
 
SO2 -> SO4
SO2 -> diabetes_prev
SO2 -> hypertension_prev

SO4 -> diabetes_prev
SO4 -> hypertension_prev

BC -> PM25
DMS -> PM25
PM -> PM25
OC -> PM25
SO2 -> PM25
SO4 -> PM25

BC -> excess
DMS -> excess
PM -> excess
OC -> excess
SO2 -> excess
SO4 -> excess

Temperature -> BC
Temperature -> DMS
Temperature -> PM
Temperature -> PM25
Temperature -> OC
Temperature -> SO2
Temperature -> SO4
Temperature -> excess
Temperature -> diabetes_prev
Temperature -> hypertension_prev

diabetes_prev -> excess
hypertension_prev -> excess
hypertension_prev -> diabetes_prev

PM25 -> diabetes_prev
PM25 -> hypertension_prev
PM25 -> excess

}')  


dagitty::coordinates(dag) <-  list(x=c(excess=0.0, PM25=-0.5, SO2=-1.6, SO4=-1.7, DMS=-1.8, OC=-1.9, PM=-2, BC=-2.1,
                                       Temperature=-1.7, diabetes_prev=-0.6, hypertension_prev=-0.4),
                                   y=c(excess=0.5, PM25=0.5, SO2=1.1, SO4=1.2, DMS=1.3, OC=1.4, PM=1.5, BC=1.6,
                                       Temperature=-0.5, diabetes_prev=2.5, hypertension_prev=2.3))




plot(dag)


## check whether any correlations are perfect (i.e., collinearity)
dataset %>% drop_na()
myCov <- cov(dataset)
round(myCov, 2)

myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)

det(myCov) < 0

any(eigen(myCov)$values < 0)


## Conditional independences
impliedConditionalIndependencies(dag)
corr <- lavCor(dataset)

#summary(corr)

localTests(dag, sample.cov=corr, sample.nobs=nrow(dataset))
#No conditional idependences


#identification
simple_dag <- dagify(
  excess ~  PM25 + SO2 +	SO4 +	DMS	+ OC +	PM +	BC	+ diabetes_prev	+ hypertension_prev + Temperature,
  PM25 ~ BC + DMS + PM + OC + SO2 + SO4  + Temperature, 
  BC ~ DMS + PM + OC + SO2 + SO4 + Temperature,
  DMS ~ PM + OC + SO2 + SO4 + Temperature,
  PM ~ OC + SO2 + SO4 + Temperature,
  OC ~ SO2 + SO4 + Temperature,
  SO2 ~ SO4 + Temperature,
  hypertension_prev ~ SO2 +	SO4 +	DMS	+ OC +	PM +	BC + PM25 + Temperature, 
  diabetes_prev ~ SO2 +	SO4 +	DMS	+ OC +	PM +	BC + PM25 + Temperature + hypertension_prev,
  
  exposure = "PM25",
  outcome = "excess",
  coords = list(x=c(excess=0.0, PM25=-0.5, SO2=-1.6, SO4=-1.7, DMS=-1.8, OC=-1.9, PM=-2, BC=-2.1,
                    Temperature=-1.7, diabetes_prev=-0.6, hypertension_prev=-0.4),
                y=c(excess=0.5, PM25=0.5, SO2=1.1, SO4=1.2, DMS=1.3, OC=1.4, PM=1.5, BC=1.6,
                    Temperature=-0.5, diabetes_prev=2.5, hypertension_prev=2.3))
)



# theme_dag() coloca la trama en un fondo blanco sin etiquetas en los ejes
ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()


#adjusting
adjustmentSets(simple_dag,  type = "minimal")
## {z_miseria, indixes}


ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()

