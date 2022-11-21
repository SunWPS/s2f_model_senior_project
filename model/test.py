from s2fgenerator.loss_function import Total_loss

total_loss = Total_loss(10).get_total_loss_func()
print(type(total_loss))