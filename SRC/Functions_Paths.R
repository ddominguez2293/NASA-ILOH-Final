# Functions_Paths.R
# Shared paths & helper functions for the whole pipeline.
# Sourced at the top of every .Rmd step with: source("SRC/Functions_Paths.R")
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code

## 1. Packages
### Load every package the R steps need (quietly)
suppressPackageStartupMessages({
  library(sf)  # spatial vectors, shp / gpkg I/O
  library(dplyr)  # data wrangling
  library(purrr)  # map_dfr() over sites
  library(nhdplusTools)  # NHDPlus HR flowlines / catchments
  library(hydroloom)  # network / upstream tracing
  library(dataRetrieval)  # USGS NWIS queries (Q and T)
  library(tidyverse)  # tidyr, ggplot2, readr, ...
  library(arrow)  # read / write parquet
  library(slider)  # rolling-window helpers
  library(lubridate)  # date handling
  library(zoo)  # time-series helpers
  library(DBI)  # database interface
  library(duckdb)  # SQL on parquet (PET.Rmd)
})

## 2. Paths
### Assumes your working directory is the repo root
base_dir <- getwd()  # project root

data_dir          <- file.path(base_dir, "data")  # all data lives under data/
shape_dir         <- file.path(data_dir, "shapes")  # Boundaries.py outputs + basin gpkg
nhd_dir           <- file.path(data_dir, "nhdplus")  # NHDPlus flowlines / catchments
meteo_dir         <- file.path(data_dir, "meteo")  # GridMET extractions
daily_drivers_dir <- file.path(data_dir, "daily_drivers")  # parquet tables passed between steps

### If directory doesn't exist it will create it otherwise move on
ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(path)
}

### AOI shapefile (Illinois River basin)
basin_shp <- file.path(shape_dir, "illinois_river_basin.shp")


### Ensure all expected directories exist
ensure_dir(data_dir)
ensure_dir(shape_dir)
ensure_dir(nhd_dir)
ensure_dir(meteo_dir)
ensure_dir(daily_drivers_dir)

### File Paths
catchment_path <- file.path(nhd_dir, "catchments.gpkg")
basins_path    <- file.path(shape_dir, "site_upstream_basins.gpkg")
flowlines_path <- file.path(nhd_dir, "flowlines.gpkg")

## 3. Helper functions
### Geopackage overwrite
#### Deletes the old file first so st_write() never appends to a stale layer
write_gpkg_overwrite <- function(obj, path) {
  if (file.exists(path)) file.remove(path)
  st_write(obj, path, quiet = TRUE)
  invisible(path)
}

### Extract an NLDI basin for one COMID (get_basin_safe)
#### Returns NULL instead of an error when the lookup fails
#### Not used by the current pipeline (upstream basins are built locally in Catchments.Rmd); kept as a utility
get_basin_safe <- purrr::possibly(
  function(cid) {
    b <- get_nldi_basin(list(featureSource = "comid", featureID = cid))
    if (is.null(b)) return(NULL)
    dplyr::mutate(b, COMID = cid)
  },
  otherwise = NULL
)

### Date cleaning for GridMET names
#### e.g., "Xpr_1984.01.01" -> "1984-01-01"

clean_date <- function(raw_name) {
  raw_name <- sub("^X", "", raw_name)           # remove leading X
  raw_name <- sub("^[A-Za-z]+_", "", raw_name)  # remove prefixes like pr_, tmmx_
  raw_name <- gsub("\\.", "-", raw_name)        # dots → hyphens
  raw_name <- trimws(raw_name)                  # remove spaces
  as.Date(raw_name, format = "%Y-%m-%d")
}

### Extract one GridMET variable to polygons (extract_variable)
#### Loops over years between start_date and end_date so a timeout / failure
#### in one year doesn't kill the whole multi-decade pull
#### Args: varname = GridMET variable, polygons = sf layer, out_dir = where <varname>_daily.csv is written, id_col = polygon ID column
#### Requires: climateR, terra, exactextractr, sf. Used by Meteo.Rmd

extract_variable <- function(varname,
                             polygons,
                             start_date,
                             end_date,
                             out_dir,
                             id_col = "comid") {
  message("Processing variable: ", varname)
  
  # Normalize dates
  start_date <- as.Date(start_date)
  end_date   <- as.Date(end_date)
  
  if (start_date > end_date) {
    stop("start_date must be <= end_date")
  }
  
  # Ensure output directory exists
  ensure_dir(out_dir)
  
  # List to accumulate per-year data
  all_years_list <- list()
  
  # Keep original polygons; we'll transform to raster CRS once
  polygons_orig <- polygons
  polygons_r    <- NULL
  rast_crs      <- NULL
  
  years <- seq(as.integer(format(start_date, "%Y")),
               as.integer(format(end_date,   "%Y")))
  
  for (yy in years) {
    # Chunk window restricted to this year and overall requested range
    chunk_start <- max(start_date, as.Date(sprintf("%d-01-01", yy)))
    chunk_end   <- min(end_date,   as.Date(sprintf("%d-12-31", yy)))
    
    if (chunk_start > chunk_end) next
    
    message("  Year ", yy, ": ", chunk_start, " to ", chunk_end)
    
    # Pull GridMET for this year
    r_list <- tryCatch(
      climateR::getGridMET(
        AOI       = polygons_orig,
        varname   = varname,
        startDate = as.character(chunk_start),
        endDate   = as.character(chunk_end)
      ),
      error = function(e) {
        message("   Skipping year ", yy, " due to error: ", e$message)
        return(NULL)
      }
    )
    
    if (is.null(r_list)) next
    
    # Convert to terra raster
    if (is.list(r_list)) {
      r <- terra::rast(r_list)
    } else {
      r <- r_list
    }
    
    # On first successful year, align polygons to raster CRS once
    if (is.null(rast_crs)) {
      rast_crs  <- terra::crs(r, proj = TRUE)
      polygons_r <- sf::st_transform(polygons_orig, rast_crs)
    }
    
    # Clean layer names into dates
    r_dates <- names(r)
    r_dates <- clean_date(r_dates)
    names(r) <- as.character(r_dates)
    
    # Extract daily means for each polygon in this year
    df_list <- lapply(seq_along(r_dates), function(i) {
      day_vals <- exactextractr::exact_extract(r[[i]], polygons_r, fun = "mean")
      tmp <- data.frame(
        id    = polygons_r[[id_col]],
        date  = r_dates[i],
        value = day_vals
      )
      names(tmp)[1] <- id_col
      tmp
    })
    
    year_df <- do.call(rbind, df_list)
    all_years_list[[length(all_years_list) + 1]] <- year_df
  }
  
  if (!length(all_years_list)) {
    warning("No data returned for variable ", varname,
            " between ", start_date, " and ", end_date)
    return(data.frame())
  }
  
  # Combine all years
  df_long <- do.call(rbind, all_years_list)
  colnames(df_long)[3] <- varname
  
  # Write per-variable CSV
  out_file <- file.path(out_dir, paste0(varname, "_daily.csv"))
  write.csv(df_long, out_file, row.names = FALSE)
  message("Saved ", varname, " to: ", out_file)
  
  df_long
}

### Download daily water temperature for one site (download_temp)
#### Uses global start_date / end_date defined in the calling script (Temp.Rmd)
#### Returns NULL on error or empty result so purrr::map_dfr() skips the site
download_temp <- function(site_id) {
  tryCatch({
    message("Downloading: ", site_id, " from ", start_date, " to ", end_date)
    
    # Strip "USGS-" prefix because NWIS expects just the site number
    site_num <- sub("USGS-", "", site_id)
    
    df <- dataRetrieval::readNWISdv(
      siteNumbers = site_num,
      parameterCd = "00010",  # water temperature (°C)
      startDate   = as.character(start_date),
      endDate     = as.character(end_date)
    )
    
    if (nrow(df) > 0) {
      df <- dataRetrieval::renameNWISColumns(df)
      df$monitoring_location_id <- site_id
      return(df)
    } else {
      return(NULL)
    }
  }, error = function(e) {
    message("Error for site ", site_id, ": ", e$message)
    return(NULL)
  })
}
