---
layout: post
title:  "AWS CloudWatch Logs Insights"
date:   2024-05-27 20:00:00 +0100
categories: other
---

In this post, some examples of the query language used by AWS CloudWatch to filter specific information from logs are shown.

-------------------------------

Filter all the logs containing the text "text contained in message":
```
fields @timestamp, @message
| filter @message like /text contained in message/
| sort @timestamp desc
| limit 90
```



Filter all the logs containing the text "[ERROR]" and extract for each occurrence the if of the event and the error type. We suppose the logs are structured as in the following example:

```
[ERROR] "app": "my_app", "delay": "440", "id_event": "224", "error_type": "unknown", "other_info": "none"
```

The query to exctract the informations follows:
```
fields @timestamp, @message
| filter @message like /[ERROR]/
| sort @timestamp desc
| parse @message '"id_event": "*", "error_type": "*"' as @id_event, @error_type
| display @id_event, @error_type, @timestamp
| limit 90
```