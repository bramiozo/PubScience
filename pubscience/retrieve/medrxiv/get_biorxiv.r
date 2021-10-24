library(medrxivr)
library(stringr)

{  
   query_interactive <- function(q){
        cat("What is the search query?: ")
        query <- readLines("stdin", n=1)
        return (query)        
    }
    query_str = query_interactive()
}

# OR c("term1", "term2")
# AND list(c1, c2)
# split by AND , then by OR
# ..append c(..) to list()
query = vector(mode = "list", length = 1)
and_list <- str_split(query_str, " AND ", simplify=FALSE)
for (and_vector in and_list){
    or_list <- str_split(and_vector, " OR ", simplify=FALSE)
    for (or_vector in or_list){
        query <- c(list(or_vector), query)
    }
}

cat(paste("Query:", query, sep=" "))

cserver_interactive <- function(){
    {
        cat("Biorxiv (type b) or Medrxiv (type m)? ")
        cserver <- readLines("stdin", n=1)
    }
    if (cserver %in% c('b', 'B'))
        return('biorxiv')
    else if (cserver %in% c('m', 'M'))
        return('medrxiv')
    else{
        cat("This is not a valid option, type 'b' for biorxiv or 'm' for medrxiv")
        cserver_interactive()
    }
}
cserver = cserver_interactive()
rootfolder = "/media/bramiozo/DATA-FAST/text_data/pubscience/biorxiv_medrxriv"
inputfolder = paste(rootfolder, "pdfdump", sep="/")

if (cserver == 'biorxiv'){
    preprint_data <- mx_api_content(server=cserver)
} else{
    preprint_data <- mx_snapshot()
}
search_query = c(query)
cat('Fetching results...')
results <- mx_search(data=preprint_data, query=search_query)
cat('Downloading pdf\'s...')
mx_download(results, inputfolder, create=FALSE)

# turn pdf's into text and extract the abstracts
# useful source: https://stackoverflow.com/questions/21445659/use-r-to-convert-pdf-files-to-text-files-for-text-mining

library(pdftools)
pdf_paths <- list.files(path=inputfolder, pattern="pdf", full.names=FALSE)
for (pdf_file in pdf_paths) {
    list_output <- pdftools::pdf_text(paste(inputfolder, pdf_file, sep="/"))
    fileConn<- file(paste(rootfolder, "articledump", paste(pdf_file, "txt", sep="."), sep="/"))
    pdf_txt <- paste(unlist(list_output),collapse=" ")
    writeLines(pdf_txt, fileConn)
    close(fileConn)

    #abstract_txt <- regmatches(pdf_txt, gregexpr("(Abstract).*(Introduction)", pdf_txt, perl=TRUE, ignore.case=TRUE))[[1]][1]
    pdf_txt <- str_replace_all(pdf_txt, "[\r\n]", " ")
    abstract_txt <- gsub(".*Abstract(.*)Introduction.*", "\\1", pdf_txt, perl=TRUE, ignore.case=TRUE)
    fileConn<-file(paste(rootfolder, "abstractdump", paste(pdf_file, "txt", sep="."), sep="/"))
    writeLines(abstract_txt, fileConn)
    close(fileConn)
}   
