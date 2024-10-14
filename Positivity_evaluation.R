# Load necessary libraries
library(ggplot2)
library(tidyverse)

data_all <- read.csv("D:/data.csv")
data <- na.omit(data_all)

# PM25 binary 
PM25_threshold <- 10
data <- data %>%
  mutate(PM25_binary = ifelse(PM25 > PM25_threshold, 1, 0))

str(data)

# Fit a logistic regression model to estimate the propensity scores
propensity_model <- glm(PM25_binary ~ BC + DMS + PM + OC + SO2 + SO4 + Temperature, 
                        data = data, family = "binomial")

# Extract propensity scores
propensity_scores <- predict(propensity_model, type = "response")

# Add propensity scores to the dataset
data <- cbind(data, propensity_score = propensity_scores)

# Check the distribution of propensity scores
summary(data$propensity_score)

# Generate new dataset
set.seed(123)
data <- data.frame(
  Treat = sample(0:1, 100, replace = TRUE),
  propensity_score = runif(100)
)

# Plot the distribution overlap
ggplot(data, aes(x = propensity_score, fill = factor(Treat))) +
  geom_density(alpha = 0.5) +
  labs(x = "Propensity Score", y = "Density", fill = "Treat") +
  ggtitle("Distribution Overlap of Propensity Score by Treatment Group")

