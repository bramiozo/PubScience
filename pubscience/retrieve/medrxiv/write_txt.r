rootfolder = "/media/bramiozo/DATA-FAST/text_data/biorxiv_medrxriv_abstracts"
inputfolder = paste(rootfolder, "pdfdump", sep="/")

library(pdftools)
library(stringr)

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
