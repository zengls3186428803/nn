a = {
    "as": 1,
    213: "ssss",
}
for k, v in a.items():
    a[k] = "%%%%" + str(v)
print(a)
