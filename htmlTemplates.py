css = '''
<style>
.chat-container {
    font-family: 'Segoe UI', sans-serif;
}

.chat-message {
    padding: 1.2rem;
    border-radius: 1rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: flex-start;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    max-width: 90%;
    word-wrap: break-word;
}

.chat-message.user {
    background-color: #1f2937;
    margin-left: auto;
    color: #e5e7eb;
}

.chat-message.bot {
    background-color: #374151;
    margin-right: auto;
    color: #f3f4f6;
}

.chat-message .avatar {
    width: 50px;
    height: 50px;
    margin-right: 1rem;
}

.chat-message.user .avatar {
    margin-left: 1rem;
    margin-right: 0;
    order: 2;
}

.chat-message .avatar img {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #9ca3af;
}

.chat-message .message {
    flex: 1;
    font-size: 1rem;
    line-height: 1.6;
    padding-top: 2px;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="asssets/robo.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
    <div class="avatar">
        <img src="asssets/human.jpg">
    </div>
</div>
'''
