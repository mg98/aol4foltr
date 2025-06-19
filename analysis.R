library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(tidyverse)
library(tikzDevice)
library(RColorBrewer)
library(lubridate)

print_plot <- function(plot, name, width=3.4, height=1.5){
  tex_name <- sprintf("results/fig/%s.tex", name)
  pdf_name <- sprintf("results/fig/%s.pdf", name)
  tex_width <- width; tex_height <- height
  pdf_width <- tex_width * 4; pdf_height <- tex_height * 4
  
  sans_preamble <- c(
    "\\usepackage{pgfplots}",
    "\\pgfplotsset{compat=newest}",
    "\\usepackage[utf8]{inputenc}",
    "\\usepackage[T1]{fontenc}",
    "\\usepackage{sfmath}",
    "\\renewcommand{\\familydefault}{\\sfdefault}"
  )
  
  tikz(file = tex_name, width = tex_width, height = tex_height, sanitize = TRUE,
       documentDeclaration = "\\documentclass[12pt]{standalone}",
       packages = sans_preamble)
  print(plot)
  dev.off()
  
  pdf(file = pdf_name, width = pdf_width, height = pdf_height)
  print(plot)
  dev.off()
}

# Load data
metadata <- read_csv("dataset/metadata.csv", 
                     col_types = cols(
                       qid = col_double(),
                       user_id = col_double(),
                       time = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
                       query = col_character(),
                       target_doc_id = col_character(),
                       candidate_doc_ids = col_character()
                     ))

##############################################
################# Statistics #################
##############################################

num_docs <- metadata %>%
  mutate(candidate_list = strsplit(candidate_doc_ids, ",")) %>%
  pull(candidate_list) %>%
  unlist() %>%
  unique() %>%
  length()

print(sprintf("#Query logs: %s", nrow(metadata)))
print(sprintf("#Docs: %s", num_docs))
print(sprintf("#Users: %s", length(unique(metadata$user_id))))
print(sprintf("#Unique queries: %s", length(unique(metadata$query))))
print(sprintf("#Unique target docs: %s", length(unique(metadata$target_doc_id))))


##############################################
############## FOLTR Experiment ##############
##############################################

results <- read_csv("results/experiment_results.csv")

results_long <- results %>%
  pivot_longer(
    cols = starts_with("mrr_"),
    names_to = "method",
    values_to = "mrr"
  ) %>%
  mutate(
    method = case_when(
      method == "mrr_real_async" ~ "Async.",
      method == "mrr_real_sync" ~ "Sync", 
      method == "mrr_iid_async" ~ "Async. (IID)",
      method == "mrr_iid_sync" ~ "Sync. (IID)"
    )
  )

method_colors <- c(
  "Async." = brewer.pal(4, "Set1")[2],
  "Sync" = brewer.pal(4, "Set1")[1],
  "Async. (IID)" = brewer.pal(4, "Set1")[4],
  "Sync. (IID)" = brewer.pal(4, "Set1")[3]
)

foltr_plot <- ggplot(results_long, aes(x = event_index, y = mrr, linetype = method, color = method)) +
  geom_line(linewidth = 0.5) +
  scale_linetype_manual(values = c("solid", "dashed", "dotted", "dotdash")) +
  scale_color_manual(values = method_colors) +
  #xlim(0, 500) +
  labs(
    x = "Rounds",
    y = "Mean Reciprocal Rank",
    linetype = "Method",
    color = "Method"
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    panel.grid.minor = element_blank(),
    legend.margin = margin(t = -10),  # Reduce top margin of legend
    plot.margin = margin(b = 5)       # Reduce bottom margin of plot
  )

print_plot(foltr_plot, "experiment", height=2.1)


##############################################
################ Query Bursts ################
##############################################

# Filter for top users and create time bins
query_timeline <- metadata %>%
  filter(user_id %in% top_users) %>%
  mutate(
    user_id = factor(user_id, levels = top_users),
    time_bin = floor_date(time, "day")
  ) %>%
  group_by(user_id, time_bin) %>%
  summarise(
    query_count = n(),
    .groups = "drop"
  )

# Create the timeline plot with each user on a horizontal line
query_bursts_plot <- ggplot(query_timeline, aes(x = time_bin, y = user_id, size = query_count)) +
  geom_point(color = "#E41A1C", alpha = 0.7) +
  scale_size_area(max_size = 4, guide = "none") +
  labs(
    x = "Timestamp",
    y = "User ID",
  ) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(linetype = "dashed", color = "grey80")
  )

print_plot(query_bursts_plot, "query_bursts")

##############################################
############# User Query Counts ##############
##############################################

user_query_counts <- metadata %>%
  group_by(user_id) %>%
  summarise(query_count = n()) %>%
  arrange(desc(query_count)) %>%
  mutate(users_with_at_least = row_number())

user_queries_plot <- ggplot(user_query_counts, aes(x = query_count, y = users_with_at_least)) +
  geom_line(color = "#377EB8", size = 1) +
  scale_x_log10(breaks = c(200, 500, 1000, 2000, 10000, 50000)) +
  scale_y_log10() +
  labs(
    x = "Number of clicks (x)",
    y = "Users with >=x clicks"
  ) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank()
  )

print_plot(user_queries_plot, "user_queries")


##############################################
############ Feature Divergence ##############
##############################################

feat_div_results <- read_csv("results/feature_divergences.csv")

feat_div_plot <- feat_div_results %>%
  ggplot(aes(x = feature_index + 1, y = divergence)) +
  geom_bar(stat = "identity", fill = "#377EB8", alpha = 0.7) +
  labs(
    x = "Feature index", 
    y = "Wasserstein distance"
  ) +
  theme_bw() +
  scale_x_continuous(breaks = seq(0, max(data$feature_index) + 1, by = 20))

print_plot(feat_div_plot, "divergence", height=1.5)
