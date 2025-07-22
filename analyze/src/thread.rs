pub struct MessageThread<'a> {
    pub root: &'a Message<'a>,
}

pub struct Message {
    pub id: String,
    // pub in_reply_to: Option<&Message>,
    // pub references: Vec<&Message>,
    pub sender: String,
    pub subject: String,
    pub children: Vec<Box<Message>>
}

