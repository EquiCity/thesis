library(dplyr)
library(ggplot2)
library(tidyr)

setwd("C:/Users/Nielja/Documents/Rico Thesis")

### ==== Load datasets
windows()

#Covid interventions
df_cov <- read.csv("./Data/CovidInterventions2020-2022_Netherlands.csv")

#Get only relevant interventions, i.e. close public transport
df_cov_t <- df_cov %>%
  select(Date, "C5_Close.public.transport", C5_Flag, C5_Notes) %>%
  mutate(Date = as.Date(as.character(Date), format = "%Y%m%d")) %>%
  mutate(PT_Closure = C5_Close.public.transport,
         C5_Notes = trimws(C5_Notes)) %>%
  select(-"C5_Close.public.transport")


#Get unique interventions
unique(df_cov_t$C5_Notes)

#Get unique statements on those days where intervention is 1
unique(df_cov_t[df_cov_t$PT_Closure == 1, "C5_Notes"])
###Outcome
#' It seems that there is only one instance where the actual schedule is affected,
#' mostly it is just recommendations for wearing face masks and avoiding peak travel
#' times, but public transport is still operational.
#' However, this intervention is also on a train level and not on local public transport.
#' It ended on the 2nd of June 2020

#Plot this over time
ggplot(df_cov_t) +
  geom_line(aes(Date, PT_Closure)) +
  geom_point(aes(Date, PT_Closure), data = df_cov_t[grepl("adjusted schedule",
                                                          df_cov_t$C5_Note, fixed = TRUE),],
  color = "red")

### ==== Look at inequalities
df_iq <- read.csv("./Data/inequalities.csv")

#Preprocessing
df_iq <- df_iq %>%
  mutate(Date = as.Date(date)) %>%
  select(-date) %>%
  #Get the different inequality measures in one column
  pivot_longer(cols = c("theil_inequality", "theil_bg_inequality", "theil_wg_inequality"),
               names_to = "inequality_metric",
  values_to = "inequality_value")

#Plot pattern of inequality over time
(g <- df_iq %>%
  filter(metric == "avg_tt") %>%
  ggplot() +
  geom_point(aes(x = Date, y = inequality_value, color = inequality_metric)))

#' Note: within group inequality is significantly higher than between groups? Total inequality
#' is dominated by wg inequality

#Add covid interventions on top
g +
  geom_line(aes(Date, PT_Closure/10), data = df_cov_t) #ugly but it is what it is

#Join the data sets and see if we see any statistical difference (unlikely, because intervention is on national level)
df_joint <- df_iq %>%
  left_join(., df_cov_t) %>%
  mutate(PT_Closure = as.factor(PT_Closure))

#Make boxplot of inequality w and wo intervention
df_joint %>%
  ggplot() +
  geom_boxplot(aes(x = PT_Closure, y = inequality_value, color = inequality_metric))
#' no difference visible

### === Load complete stats
df_stats <- read.csv("./Data/complete_stats.csv")

#Pivot to longer form and see if there was any change in avg_nodes, and avg_lines per buurt
df_stats <- df_stats %>%
  pivot_longer(cols = contains("BU"),
               names_to = "buurt_ID",
  values_to = "values") %>%
  mutate(Date = as.Date(date)) %>%
  select(-date) %>%
  left_join(., df_cov_t) %>%
    left_join(., df_iq)

#Check which metrics we have
unique(df_stats$metric)

#Make some plots -> NOTE: This takes a while!
df_stats %>%
  ggplot() +
  geom_line(aes(x = Date, y = values, color = buurt_ID), alpha = 0.2) +
  facet_wrap(factor(metric)~., scales = "free_y") +
  guides(color = "none") +
  theme_light()

