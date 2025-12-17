# validation_against_StMoMo/stmomo_fit_lc.R
# ------------------------------------------------------------
# Fit Lee–Carter (LC) with StMoMo on France Excel dataset (3 sheets):
#   - "Death Rate"
#   - "Exposure to risk"
#   - "Deaths"
#
# Data file:
#   Data/data_france.xlsx
#
# Outputs:
#   validation_against_StMoMo/outputs/
#     lc_params.csv
#     lc_fitted_logm.csv
#     lc_kt.csv
#     lc_data_summary.json
# ------------------------------------------------------------

if ("package:demography" %in% search()) detach("package:demography", unload = TRUE)

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(tibble)
})

# -----------------------------
# Config
# -----------------------------
ROOT_DIR <- "."  # run from repo root
DATA_DIR <- file.path(ROOT_DIR, "Data")
OUT_DIR  <- file.path(ROOT_DIR, "validation_against_StMoMo", "outputs")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

XLSX_PATH <- file.path(DATA_DIR, "data_france.xlsx")

# Choose which column to use: "Total", "Male", or "Female"
SEX_COL <- "Total"

# Match PYMORT lifetables.py (you said you use 60..100)
AGE_MIN  <- 60
AGE_MAX  <- 100

# Optional year bounds (leave NULL to keep all available after filtering)
YEAR_MIN <- 1970
YEAR_MAX <- 2019

TARGET_AGES <- seq(AGE_MIN, AGE_MAX, by = 1)
TARGET_YEARS <- if (!is.null(YEAR_MIN) && !is.null(YEAR_MAX)) {
  seq(YEAR_MIN, YEAR_MAX, by = 1)
} else {
  NULL
}

# -----------------------------
# Helpers
# -----------------------------
stop_if_missing_cols <- function(df, cols, sheet_name) {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    stop(sprintf("Sheet '%s' is missing columns: %s",
                 sheet_name, paste(missing, collapse = ", ")))
  }
}

clean_age <- function(x) {
  # Handles: "100+", " 60 ", "60 years", etc.
  x <- as.character(x)
  x <- gsub("\\+", "", x)
  x <- gsub("[^0-9\\.]", "", x)
  suppressWarnings(as.numeric(x))
}

read_sheet_long <- function(path, sheet) {
  df <- read_excel(path, sheet = sheet, skip = 2)
  df <- as.data.frame(df)
  names(df) <- trimws(names(df))

  req <- c("Year", "Age", "Female", "Male", "Total")
  stop_if_missing_cols(df, req, sheet)

  df %>%
    mutate(
      Year = suppressWarnings(as.integer(Year)),
      Age  = clean_age(Age)
    ) %>%
    filter(!is.na(Year), !is.na(Age)) %>%
    select(Year, Age, Female, Male, Total)
}

apply_filters <- function(df) {
  out <- df %>%
    filter(Age >= AGE_MIN, Age <= AGE_MAX)

  if (!is.null(YEAR_MIN)) out <- out %>% filter(Year >= YEAR_MIN)
  if (!is.null(YEAR_MAX)) out <- out %>% filter(Year <= YEAR_MAX)

  out
}

# Convert joined long df (Year, Age, value) -> matrix [age, year]
to_matrix_age_year <- function(df, value_col, ages, years) {
  wide <- df %>%
    select(Year, Age, !!sym(value_col)) %>%
    tidyr::pivot_wider(names_from = Year, values_from = !!sym(value_col))

  wide <- wide %>%
    arrange(Age)

  # Ensure every requested year column exists (even if missing -> NA)
  for (yy in years) {
    yname <- as.character(yy)
    if (!(yname %in% names(wide))) wide[[yname]] <- NA_real_
  }

  wide <- wide %>% select(Age, all_of(as.character(years)))

  mat <- as.matrix(wide %>% select(-Age))
  storage.mode(mat) <- "double"
  rownames(mat) <- as.character(wide$Age)
  colnames(mat) <- as.character(years)
  mat
}

build_stmomo_data <- function(Dxt, Ext, ages, years, type, series_label) {
  attempt <- tryCatch(
    StMoMo::StMoMoData(
      Dxt = Dxt,
      Ext = Ext,
      ages = ages,
      years = years,
      type = type
    ),
    error = identity
  )

  if (inherits(attempt, "StMoMoData")) {
    return(attempt)
  }

  # Fallback for StMoMo versions that only accept demogdata input
  manual <- list(
    Dxt = Dxt,
    Ext = Ext,
    ages = ages,
    years = years,
    ages.lab = as.character(ages),
    years.lab = as.character(years),
    type = type,
    series = series_label,
    label = "constructed_from_matrices"
  )
  class(manual) <- "StMoMoData"
  manual
}

# -----------------------------
# Load data
# -----------------------------
if (!file.exists(XLSX_PATH)) {
  stop(sprintf("Data file not found: %s", XLSX_PATH))
}

death_rate_df <- read_sheet_long(XLSX_PATH, "Death Rate")       %>% apply_filters()
exposure_df   <- read_sheet_long(XLSX_PATH, "Exposure to risk") %>% apply_filters()
deaths_df     <- read_sheet_long(XLSX_PATH, "Deaths")           %>% apply_filters()

# Rename columns so we can join safely
joined <- death_rate_df %>%
  rename(mx_Female = Female, mx_Male = Male, mx_Total = Total) %>%
  inner_join(
    exposure_df %>% rename(Ex_Female = Female, Ex_Male = Male, Ex_Total = Total),
    by = c("Year", "Age")
  ) %>%
  inner_join(
    deaths_df %>% rename(Dx_Female = Female, Dx_Male = Male, Dx_Total = Total),
    by = c("Year", "Age")
  )

sex_map <- list(
  "Female" = list(mx = "mx_Female", Ex = "Ex_Female", Dx = "Dx_Female"),
  "Male"   = list(mx = "mx_Male",   Ex = "Ex_Male",   Dx = "Dx_Male"),
  "Total"  = list(mx = "mx_Total",  Ex = "Ex_Total",  Dx = "Dx_Total")
)
if (!(SEX_COL %in% names(sex_map))) stop("SEX_COL must be one of: 'Female', 'Male', 'Total'")

mx_col <- sex_map[[SEX_COL]]$mx
Ex_col <- sex_map[[SEX_COL]]$Ex
Dx_col <- sex_map[[SEX_COL]]$Dx

# Keep only needed columns + drop impossible rows for Poisson LC:
# StMoMo requires Ext > 0 and finite everywhere we fit.
joined2 <- joined %>%
  transmute(
    Year = Year,
    Age  = Age,
    mx   = as.numeric(.data[[mx_col]]),
    Ex   = as.numeric(.data[[Ex_col]]),
    Dx   = as.numeric(.data[[Dx_col]])
  ) %>%
  filter(
    Age %in% TARGET_AGES,
    is.finite(Year), is.finite(Age),
    is.finite(Ex), is.finite(Dx),
    Ex > 0,
    Dx >= 0
  )

# Build a complete rectangular grid based on remaining data
ages  <- sort(unique(joined2$Age))
years <- sort(unique(joined2$Year))
if (!is.null(TARGET_YEARS)) {
  years <- intersect(years, TARGET_YEARS)
}

# Force exactly ages 60..100 if present; otherwise use intersection
ages <- intersect(TARGET_AGES, ages)

# Filter to those ages/years only
joined2 <- joined2 %>% filter(Age %in% ages, Year %in% years)

# Recompute years after filtering
years <- sort(unique(joined2$Year))
if (!is.null(TARGET_YEARS)) {
  years <- intersect(years, TARGET_YEARS)
}

# Build matrices
Ext <- to_matrix_age_year(joined2, "Ex", ages, years)
Dxt <- to_matrix_age_year(joined2, "Dx", ages, years)

# Drop any age rows that still contain NA in Ext/Dx across years
# (so Ext is strictly positive everywhere on the kept grid)
good_age <- apply(Ext, 1, function(r) all(is.finite(r)) && all(r > 0))
good_age <- good_age & apply(Dxt, 1, function(r) all(is.finite(r)) && all(r >= 0))

Ext <- Ext[good_age, , drop = FALSE]
Dxt <- Dxt[good_age, , drop = FALSE]
ages_fit <- as.numeric(rownames(Ext))

# Drop any year columns that contain NA in Ext/Dx
good_year <- apply(Ext, 2, function(c) all(is.finite(c)) && all(c > 0))
good_year <- good_year & apply(Dxt, 2, function(c) all(is.finite(c)) && all(c >= 0))

Ext <- Ext[, good_year, drop = FALSE]
Dxt <- Dxt[, good_year, drop = FALSE]
years_fit <- as.integer(colnames(Ext))

if (length(ages_fit) < 5) stop("Too few ages left after filtering. Check your sheet values / parsing.")
if (length(years_fit) < 5) stop("Too few years left after filtering. Check your sheet values / parsing.")

# Final safety checks on inputs to StMoMo
if (!is.matrix(Ext) || !is.numeric(Ext)) stop("Ext must be a numeric matrix")
if (!is.matrix(Dxt) || !is.numeric(Dxt)) stop("Dxt must be a numeric matrix")
if (anyNA(Ext) || anyNA(Dxt)) stop("Ext/Dxt contain NA after filtering")
if (any(Ext <= 0)) stop("Ext must be strictly positive")
if (any(Dxt < 0)) stop("Dxt must be non-negative")
if (any(!ages_fit %in% TARGET_AGES)) stop("ages_fit must be within the target age range")

# -----------------------------
# Fit Lee–Carter with StMoMo
# -----------------------------
StMoMo_dat <- build_stmomo_data(
  Dxt = Dxt,
  Ext = Ext,
  ages = ages_fit,
  years = years_fit,
  type = "central",
  series_label = SEX_COL
)
stopifnot(inherits(StMoMo_dat, "StMoMoData"))

LCmod <- StMoMo::lc()

fit_lc <- StMoMo::fit(
  LCmod,
  data = StMoMo_dat,
  ages.fit = ages_fit,
  years.fit = years_fit
)

ax <- as.numeric(fit_lc$ax); names(ax) <- ages_fit
bx <- as.numeric(fit_lc$bx); names(bx) <- ages_fit
kt <- as.numeric(fit_lc$kt); names(kt) <- years_fit

# Rebuild fitted log m_{x,t} = ax_x + bx_x * kt_t
logm_hat <- matrix(NA_real_, nrow = length(ages_fit), ncol = length(years_fit))
for (i in seq_along(ages_fit)) {
  logm_hat[i, ] <- ax[i] + bx[i] * kt
}
rownames(logm_hat) <- as.character(ages_fit)
colnames(logm_hat) <- as.character(years_fit)

# -----------------------------
# Save outputs
# -----------------------------
params_df <- data.frame(
  Age = ages_fit,
  ax  = ax,
  bx  = bx
)
write.csv(params_df, file.path(OUT_DIR, "lc_params.csv"), row.names = FALSE)

logm_df <- as.data.frame(logm_hat) %>%
  rownames_to_column("Age") %>%
  mutate(Age = as.numeric(Age)) %>%
  pivot_longer(-Age, names_to = "Year", values_to = "logm_fitted") %>%
  mutate(Year = as.integer(Year))
write.csv(logm_df, file.path(OUT_DIR, "lc_fitted_logm.csv"), row.names = FALSE)

kt_df <- data.frame(Year = years_fit, kt = kt)
write.csv(kt_df, file.path(OUT_DIR, "lc_kt.csv"), row.names = FALSE)

summary <- list(
  file = XLSX_PATH,
  sex = SEX_COL,
  ages_requested = list(min = AGE_MIN, max = AGE_MAX),
  ages_used = list(min = min(ages_fit), max = max(ages_fit), n = length(ages_fit)),
  years_used = list(min = min(years_fit), max = max(years_fit), n = length(years_fit)),
  notes = "StMoMo LC fit using Dxt/Ext from sheets. Grid filtered to ensure Ext>0 everywhere."
)
writeLines(
  jsonlite::toJSON(summary, pretty = TRUE, auto_unbox = TRUE),
  file.path(OUT_DIR, "lc_data_summary.json")
)

message("Done. Outputs written to: ", OUT_DIR)
message(sprintf("Used ages %d..%d (n=%d), years %d..%d (n=%d)",
                min(ages_fit), max(ages_fit), length(ages_fit),
                min(years_fit), max(years_fit), length(years_fit)))
