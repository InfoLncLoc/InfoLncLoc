
library(pheatmap)
data = read.table("4.txt",header = T,row.names = 1,sep="\t",check.names = F,quote ="" )

log_data<- -log10(data)
range(log_data)
pdf("test2278.pdf")

bk <- c(seq(0,-log10(0.05)-0.1,by=0.001),seq(-log10(0.05),56,by=0.05))

pheatmap(log_data,
        
         color = c(colorRampPalette(colors = c("LightSkyBlue","white"))(1203),colorRampPalette(colors = c("white","orange"))(1093)),
         legend_breaks =seq(0,56,10),
         breaks = bk,
              number_color="red",number_format="%.2e",
              fontsize_row = 5,
              border="#8B0A50",
              display_numbers = matrix(ifelse(log_data > -log10(0.05), "+", ""), nrow(data)),
              cutree_cols = 3,cutree_rows =4,
              angle_col = 45,
              show_rownames = T,
              cluster_cols = F,
              cluster_rows = F,)
dev.off()

