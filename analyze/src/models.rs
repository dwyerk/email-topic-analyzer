use std::{cell::RefCell, rc::Rc};


pub struct Message<'a> {
    pub id: String,
    pub in_reply_to: Option<&'a Message<'a>>,
    pub references: Vec<&'a Message<'a>>,
    pub sender: String,
    pub subject: String,
}

impl<'a> PartialEq for MessageThread<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.message.id == other.message.id
    }
}

pub struct MessageThread<'a> {
    pub parent: Option<&'a MessageThread<'a>>,
    pub message: Message<'a>,
    pub children: Vec<&'a Rc<RefCell<MessageThread<'a>>>>,
}

impl MessageThread<'_> {
    pub fn new<'a>(message: Message<'a>) -> MessageThread<'a> {
        MessageThread {
            parent: None,
            message,
            children: vec![],
        }
    }
    pub fn add_child<'a>(&'a mut self, child: &Rc<RefCell<MessageThread<'a>>>) {
        match self.children.contains(&child) {
            true => (),
            false => self.children.push(child),
        }
    }
}