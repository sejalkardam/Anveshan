from Tools.pynche.pyColorChooser import save
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from cropyield1 import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.utils.encoding import force_str
# from . tokens import generate_token
from django.contrib import messages
from django.core.mail import EmailMessage, send_mail

from cropyield2.tokens import generate_token


def home(request):
    return render(request, "cropyield2/index.html")

def signup(request):
    if request.method=='POST':
        username=request.POST.get('username')
        fname=request.POST.get('fname')
        lname=request.POST.get('lname')
        email=request.POST.get('email')
        pass1=request.POST.get('pass1')
        pass2=request.POST.get('pass2')
        
        if User.objects.filter(username=username):
            messages.error(request, "Username Already Exist Try a Different Username !!!")
            return redirect('home')
        
        if User.objects.filter(email=email):
            messages.error(request, "Email  Already Exist Try a Different Email !!!")
            return redirect('home')
        
        if len(username)>15:
            messages.error(request, "Username Must Be Under 15 Character")
        
        if pass1!=pass2:
            messages.error(request, "Password Did't Match !!!")
            
        if not username.isalnum():
            messages.error(request, "Username Must be Alpha-Numberic !!!")
            return redirect('home')
        
            
        myuser=User.objects.create_user(username,email,pass1)
        myuser.first_name=fname
        myuser.last_name=lname 
        myuser.is_active=False
        myuser.save()
        
        messages.success(request,"Your Account is Succesfully Created !!! \n We Have Sent You a Confirmation Email , Please Verify it !!!")
        
        #Welcome to Email
        subject="Welcome to Crop Yield Prediction App"
        message= "Hello "+myuser.first_name+" "+myuser.last_name+ "\n Welcome to Crop Yield Prediction App \n We have sent you an email to confirm your email id \n Please Confirm your Email in Order to Activate Your Email Id \n\n Thank You\n"+myuser.username
        from_email=settings.EMAIL_HOST_USER
        to_list=[myuser.email]
        send_mail(subject, message, from_email, to_list,fail_silently=True)
        
        
        #Email Address Confirmation
        current_site=get_current_site(request)
        email_subject="Confirm Your Email Please !!!"
        message2=render_to_string('email_confirmation.html',{
            'name':myuser.first_name,
            'domain':current_site.domain,
            'uid':urlsafe_base64_encode(force_bytes(myuser.pk)),
            'token':generate_token.make_token(myuser)
               
        })
        
        email=EmailMessage(
            email_subject,
            message2,
            settings.EMAIL_HOST_USER,
            [myuser.email],
        )
        
        email.fail_silently=True
        email.send()
        
        return redirect('signin')
        
        
        
        
    return render(request, "cropyield2/signup.html")

def signin(request):
    if request.method=="POST":
        username=request.POST.get('username')
        pass1=request.POST.get('pass1')
        user=authenticate(username=username,password=pass1)
        
        if user is not None:
            login(request,user)
            fname=user.first_name
            return render(request,"cropyield2/information.html",{'fname':fname})
        
        else:
            messages.error(request,"Credential Not Matched !!!")
            return redirect('home')
        
        
        
    
    return render(request,"cropyield2/signin.html")

def signout(request):
   logout(request)
   messages.success(request,"Log out Succesfully!!!")
   return redirect('home')

def activate(request,uidb64,token):
    try:
        uid=force_str(urlsafe_base64_decode(uidb64))
        myuser=User.objects.get(pk=uid)
    except (TypeError,ValueError,OverflowError,User.DoesNotExist):
        myuser=None
    
    if myuser is not None and generate_token.check_token(myuser, token):
        myuser.is_active= True
        myuser.save()
        login(myuser,save)
        return redirect('home')
    else:
        return render(request,'activation_failed.html')

