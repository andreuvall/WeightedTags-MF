# Generate dummy dataset with 200 users, 500 items and 50 tags
library(data.table)
set.seed(1)

# Matrix of user-item-playcounts (200 x 500) ----
user_item = data.table(user = sample(0:199, size=1000, replace=TRUE), 
                       item = sample(0:499, size=1000, replace=TRUE),
                       count = sample(1:200, size=1000, replace=TRUE))
user_item = user_item[, list(count = sum(count)), by = list(user, item)]
setkey(user_item, user, item)

# Matrix of user-tag-counts (200 x 50) ----
user_tag = data.table(user = sample(0:199, size=100, replace=TRUE), 
                      tag = sample(0:49, size=100, replace=TRUE),
                      count = sample(1:25, size=100, replace=TRUE))
user_tag = user_tag[, list(count = sum(count)), by = list(user, tag)]
setkey(user_tag, user, tag)

# Matrix of item-tag-counts (500 x 50) ----
item_tag = data.table(item = sample(0:499, size=500, replace=TRUE), 
                      tag = sample(0:49, size=500, replace=TRUE),
                      count = sample(1:25, size=500, replace=TRUE))
item_tag = item_tag[, list(count = sum(count)), by = list(item, tag)]
setkey(item_tag, item, tag)

# Save to files ----
dir.create("./dummy_collection")
dir.create("./dummy_collection/dummy_dataset")
write.table(user_item, file="./dummy_collection/dummy_dataset/playcounts.txt", sep=",", row.names=FALSE, col.names=FALSE)
write.table(user_tag, file="./dummy_collection/dummy_dataset/user_tags.txt", sep=",", row.names=FALSE, col.names=FALSE)
write.table(item_tag, file="./dummy_collection/dummy_dataset/item_tags.txt", sep=",", row.names=FALSE, col.names=FALSE)
